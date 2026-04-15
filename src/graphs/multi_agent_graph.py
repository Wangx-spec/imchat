import json
import logging
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from agents.agent_registry import all_agents
from llms.openai_chat import build_openai_chat_model

from langgraph.prebuilt import create_react_agent
import re
from actions.basic_tools import get_actions
from prompts.system_prompts import build_system_prompt


logger = logging.getLogger(__name__)

def _rule_route(latest_user: str) -> str | None:
    text = (latest_user or "").strip().lower()

    if not text:
        return None

    if any(x in text for x in ["几点", "几号", "星期", "日期", "时间"]):
        return "chat"

    if any(x in text for x in ["怎么做", "做法", "步骤", "食材", "需要哪些食材", "需要什么材料"]):
        return "knowledge"

    if any(x in text for x in ["推荐几道", "推荐几个", "有什么菜", "想吃啥", "适合晚餐", "高难度", "简单点的菜"]):
        return "recommend"

    if re.search(r"[\d\(\)\+\-\*/\.]+", text) and any(x in text for x in ["等于多少", "计算", "+", "-", "*", "/"]):
        return "chat"

    return None

def _preview_latest_user_message(state: MessagesState, max_chars: int = 80) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", "") == "human":
            content = getattr(msg, "content", "")
            text = content if isinstance(content, str) else str(content)
            text = text.strip().replace("\n", " ")
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text
    return ""

def _build_supervisor_node(llm, agent_defs):
    agent_descriptions = "\n".join(
        f'- "{a.name}": {a.description}'
        for a in agent_defs
    )
    agent_names = [a.name for a in agent_defs]

    router_prompt = f"""\
你是一个路由器，负责将用户消息分派给合适的专家处理。

可用专家：
{agent_descriptions}

规则：
1. 分析用户最新一条消息的意图
2. 选择最匹配的专家
3. 只返回 JSON: {{"agent": "<专家名>", "reason": "<一句话理由>"}}
4. 不要回答用户的问题，只做路由判断

可选的 agent 值: {json.dumps(agent_names)}"""

    def supervisor_node(state: MessagesState) -> dict:
        latest_user = _preview_latest_user_message(state)
        logger.info("[SUPERVISOR] start latest_user=%r", latest_user)
        rule_choice = _rule_route(latest_user)
        if rule_choice:
            logger.info("[SUPERVISOR] rule_routed_to=%s latest_user=%r", rule_choice, latest_user)
            return {"next": rule_choice}
        messages = [
            SystemMessage(content=router_prompt),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        try:
            parsed = json.loads(response.content)
            chosen = parsed.get("agent", agent_names[-1])
            if chosen not in agent_names:
                logger.warning(
                    "[SUPERVISOR] invalid_agent=%r fallback=%s raw=%r",
                    chosen,
                    agent_names[-1],
                    response.content,
                )
                chosen = agent_names[-1]
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "[SUPERVISOR] parse_failed fallback=%s raw=%r",
                agent_names[-1],
                getattr(response, "content", response),
            )
            chosen = agent_names[-1]

        logger.info("[SUPERVISOR] routed_to=%s raw=%r", chosen, response.content)
        return {"next": chosen}

    return supervisor_node


def _build_specialist_node(name, llm, agent_def):
    tools = get_actions(agent_def.skills)
    prompt = build_system_prompt(agent_def.skills)
    tool_names = [getattr(tool, "name", str(tool)) for tool in tools]
    logger.info(
        "[SPECIALIST_BUILD] name=%s skills=%s tools=%s",
        name,
        agent_def.skills,
        tool_names,
    )
    sub_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )

    def specialist_node(state: MessagesState) -> dict:
        latest_user = _preview_latest_user_message(state)
        logger.info("[SPECIALIST_RUN] name=%s start latest_user=%r", name, latest_user)
        result = sub_agent.invoke({"messages": state["messages"]})
        logger.info(
            "[SPECIALIST_RUN] name=%s done message_count=%d",
            name,
            len(result.get("messages", [])),
        )
        return {"messages": result["messages"]}

    return specialist_node

def build_multi_agent_graph(settings, checkpointer=None):
    llm = build_openai_chat_model(settings)
    agent_defs = all_agents()
    agent_names = [a.name for a in agent_defs]
    logger.info(
        "[MULTI_AGENT_BUILD] agent_names=%s checkpointer=%s",
        agent_names,
        type(checkpointer).__name__ if checkpointer is not None else "None",
    )

    supervisor = _build_supervisor_node(llm, agent_defs)
    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor)

    for agent_def in agent_defs:
        node = _build_specialist_node(agent_def.name, llm, agent_def)
        builder.add_node(agent_def.name, node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", agent_names[-1]),
        {name: name for name in agent_names},
    )

    for name in agent_names:
        builder.add_edge(name, END)

    return builder.compile(checkpointer=checkpointer)