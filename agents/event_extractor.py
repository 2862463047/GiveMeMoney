"""
小说事件提取模块。

将小说文本拆解为一系列按时间顺序排列的事件 (Event)
每个事件包含描述、角色、因果关系和完整的过程链。
"""

import logging
import os
import asyncio
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from interfaces import Event

# 系统提示词：定义 LLM 角色及事件提取的任务、输入格式、输出格式和指南。
system_prompt_template_extract_events = \
"""
You are a highly skilled Literary Analyst AI. Your expertise is in narrative structure, plot deconstruction, and thematic analysis. You meticulously read and interpret prose to break down a story into its fundamental sequential events.

**TASK**
Extract the next event from the provided novel, following the sequence of the story and building upon the partially extracted events.

**INPUT**
1. The full text of the novel, which is enclosed within <NOVEL_TEXT_START> and <NOVEL_TEXT_END> tags
2. A sequence of already-extracted events (in order), which is enclosed within <EXTRACTED_EVENTS_START> and <EXTRACTED_EVENTS_END> tags. The sequence may be empty. Each event contains multiple processes and constitutes a complete causal chain.

Below is an example input:

<NOVEL_TEXT_START>
The night was as dark as ink when the piercing alarm of the city museum suddenly shattered the silence. A thief, moving with phantom-like agility, had just pried open the display case and snatched the blue gem known as the "Heart of the Ocean" when the blaring alarm echoed through the hall.
... (more novel text) ...
<NOVEL_TEXT_END>

<EXTRACTED_EVENTS_START>
<Event 0>
Description: A thief who stole a gem from a museum was caught after a rooftop chase with guards, and the gem was recovered.
Process Chain:
- A thief steals a gem from a museum, triggering the alarm. Guards notice and begin the chase.
- The thief rushes out the museum's back door and dashes through narrow alleys, with guards closely pursuing and calling for backup.
- ... (more processes) ...

<Event 1>
Description: ... (more description) ...
Process Chain:
- ... (more processes) ...

<EXTRACTED_EVENTS_END>


**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Focus on events that are critical to the plot, character development, or thematic depth.
2. Ensure the event is logically distinct from previous and subsequent events.
3. If the event spans multiple scenes, unify them under a single dramatic goal. For example, a chase sequence might begin in a city market, continue through back alleys, and conclude on a rooftop—all comprising a single event because they collectively achieve the dramatic purpose of "the protagonist evading capture."
4. Maintain objectivity: describe events based on the text without interpretation or judgment.
5. For the process field, provide a detailed, step-by-step account of the event's progression, including key actions, decisions, and turning points. Each step should be clear and concise, illustrating how the event unfolds over time.
Below is an example:
Timeframe: The following morning, after acquiring the information about the Temple.
Characters: Elara (protagonist) and Kaelen (her rival treasure hunter).
Cause: Both seek the same artifact and are determined to reach it first.
Process: The event begins with Elara hastily purchasing supplies in the port town (scene 1), where she spots Kaelen already hiring a crew, raising the stakes. It continues as she races to secure her own ship and captain, negotiating fiercely under time pressure (scene 2). The event culminates in a direct confrontation on the docks (scene 3), where Kaelen attempts to sabotage her vessel, leading to a brief but intense sword fight between the two rivals.
Outcome: Elara successfully defends her ship and sets sail, but the conflict solidifies a bitter personal rivalry with Kaelen, ensuring their race to the temple will be fraught with direct opposition and danger.
6. Every detail in your event description must be directly supported by the input novel. Do not add, assume, or invent any information.
7. The language of outputs in values should be same as the input text.
"""

# 人类消息模板：将小说全文和已提取事件列表注入到提示词中。
human_prompt_template_extract_next_event = \
"""
<NOVEL_TEXT_START>
{novel_text}
<NOVEL_TEXT_END>

<EXTRACTED_EVENTS_START>
{extracted_events}
<EXTRACTED_EVENTS_END>
"""


class EventExtractor:
    """
    小说事件提取器。

    通过 LLM 逐事件地提取小说中的关键情节事件。
    """

    def __init__(
            self,
            api_key: str,
            base_url: str,
            chat_model: str,
    ):
        """
        初始化事件提取器。

        参数:
            api_key: LLM API 密钥。
            base_url:  基础 URL。
            chat_model: 模型名称
        """
        self.chat_model = init_chat_model(
            model=chat_model,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url
        )
        self.parser = PydanticOutputParser(pydantic_object=Event)

    def __call__(
            self,
            novel_text: str,
    ):
        """
        从小说文本中提取所有事件。

        循环调用 extract_next_event，每次提取一个事件，直到返回的事件
        标记 is_last=True 时停止。

        参数:
            novel_text: 小说全文文本。

        返回:
            List[Event]: 按时间顺序排列的事件列表。
        """
        logging.info("Extracting events from novel...")

        events = []
        while True:  # 循环提取事件，终止条件由 LLM 返回的 event.is_last 字段控制。
            event = self.extract_next_event(novel_text, events)

            events.append(event)
            logging.info(f"Extracted event: \n{event}")
            # 注意: 此处访问列表 events 的 is_last 属性，实际应为 event.is_last。
            if events.is_last:
                break

        return events

    @retry(  # 失败时最多重试 3 次，每次重试前记录警告日志。
        stop=stop_after_attempt(3),
        after=lambda retry_state: logging.warning(
            f"Retrying extract_next_event due to error: {retry_state.outcome.exception()}"),
    )
    def extract_next_event(
            self,
            novel_text: str,
            extracted_events: List[Event]
    ) -> Event:
        """
        提取下一个事件。

        将小说文本和已提取的事件列表拼接为 prompt，调用 LLM 提取下一个事件，
        并通过 PydanticOutputParser 解析为 Event 对象。

        参数:
            novel_text: 小说全文文本。
            extracted_events: 已提取的事件列表，作为提取下一个事件的上下文。

        返回:
            Event: 新提取的事件，其 index 与 extracted_events 的长度一致。
        """
        # 将已提取的事件列表序列化为字符串，用于填入 prompt 模板。
        extracted_events_str = "\n\n".join([str(e) for e in extracted_events])

        messages = [
            SystemMessage(
                content=system_prompt_template_extract_events.format(format_instructions=self.parser.get_format_instructions())
            ),
            HumanMessage(
                content=human_prompt_template_extract_next_event.format(
                    novel_text=novel_text,
                    extracted_events=extracted_events_str,
                )
            )
        ]

        # 通过 LCEL (LangChain Expression Language) 管道将 chat_model 的输出
        # 直接传入 PydanticOutputParser，自动解析为 Event 对象。
        chain = self.chat_model | self.parser

        event: Event = chain.invoke(messages)

        # 校验 LLM 返回的事件索引与已提取事件数量一致，确保事件顺序正确。
        assert event.index == len(
            extracted_events), f"Extracted event index {event.index} does not match the expected index {len(extracted_events)}"

        return event


