"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
import unicodedata
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""
    def query(self, user_input: str) -> str:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text so simple intent matching works with Vietnamese input."""
        return "".join(
            char
            for char in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(char) != "Mn"
        )

    def _build_actions(self, user_input: str):
        """Create the deterministic actions used by this offline lab agent."""
        normalized = self._normalize(user_input)
        airport_codes = [
            code
            for code in re.findall(r"\b[A-Z]{3}\b", user_input.upper())
            if code in {"HAN", "SGN", "DAD"}
        ]
        has_flight_request = (
            "chuyen bay" in normalized
            or "flight" in normalized
            or ("ve" in normalized and ("tu" in normalized or "den" in normalized))
        )
        has_weather_request = "thoi tiet" in normalized or "weather" in normalized
        actions = []

        if has_flight_request and len(airport_codes) >= 2:
            price_match = re.search(
                r"(\d+(?:[.,]\d+)?)\s*(?:trieu|million)", normalized
            )
            if price_match:
                max_price = int(float(price_match.group(1).replace(",", ".")) * 1_000_000)
            else:
                number_match = re.search(r"(\d[\d.,]*)\s*(?:vnd|dong)", normalized)
                max_price = int(number_match.group(1).replace(".", "").replace(",", "")) if number_match else 5_000_000

            actions.append(
                {
                    "name": "get_flight_info",
                    "args": {
                        "origin": airport_codes[0],
                        "destination": airport_codes[1],
                        "max_price": max_price,
                    },
                }
            )

        if has_weather_request and airport_codes:
            actions.append(
                {
                    "name": "get_weather_forecast",
                    "args": {"city_code": airport_codes[-1]},
                }
            )

        return actions

    @staticmethod
    def _format_answer(user_input: str, observations) -> str:
        """Turn tool observations into a concise customer-facing answer."""
        parts = []
        for observation in observations:
            if isinstance(observation, list):
                if not observation:
                    parts.append("Không tìm thấy chuyến bay phù hợp.")
                    continue
                flights = "; ".join(
                    f"{flight['flight_number']} ({flight['airline']}, "
                    f"{flight['price_vnd']:,} VND, khởi hành {flight['departure_time']})"
                    for flight in observation
                )
                parts.append(f"Các chuyến bay phù hợp: {flights}.")
            elif isinstance(observation, dict) and "city" in observation:
                parts.append(
                    f"Thời tiết tại {observation['city']}: "
                    f"{observation['temperature_c']}°C, {observation['condition']}. "
                    f"{observation['recommendation']}"
                )
            elif isinstance(observation, dict) and "error" in observation:
                parts.append(f"Không thể tra cứu: {observation['error']}.")

        if parts:
            return " ".join(parts)
        return (
            "Chính sách đổi trả vé Vinpearl phụ thuộc vào điều kiện của từng loại vé. "
            "Vui lòng kiểm tra điều kiện vé hoặc liên hệ bộ phận hỗ trợ để được tư vấn."
        )

    def run(self, user_input: str) -> str:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        # TODO 3: Phân tích Thought / Action từ Agent
        # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
        # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer

        self.trace = []
        actions = self._build_actions(user_input)
        observations = []
        action_index = 0
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            if action_index < len(actions):
                action_definition = actions[action_index]
                action_index += 1
                thought = "Cần dùng công cụ để lấy dữ liệu chính xác cho yêu cầu này."

                # Serialize and parse the action to enforce the documented JSON format.
                action_json = json.dumps(action_definition, ensure_ascii=False)
                action = action_definition
                try:
                    action = json.loads(action_json)
                    tool_name = str(action["name"]).strip().lower()
                    tool_args = action.get("args", {})
                    tool = TOOL_MAP[tool_name]
                    observation = tool(**tool_args)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    observation = {"error": f"Invalid action: {error}"}
                except Exception as error:
                    observation = {"error": str(error)}

                observations.append(observation)
                self.trace.append(
                    {
                        "iteration": iteration,
                        "thought": thought,
                        "action": action,
                        "observation": observation,
                    }
                )

                # A one-tool request can be answered in the same ReAct step.
                if len(actions) == 1:
                    answer = self._format_answer(user_input, observations)
                    self.trace[-1]["final_answer"] = answer
                    return {
                        "status": "completed",
                        "iterations": iteration,
                        "answer": answer,
                        "trace": self.trace,
                    }
                continue

            answer = self._format_answer(user_input, observations)
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Đã có đủ thông tin để trả lời người dùng.",
                    "action": None,
                    "observation": None,
                    "final_answer": answer,
                }
            )
            return {
                "status": "completed",
                "iterations": iteration,
                "answer": answer,
                "trace": self.trace,
            }

        return {
            "status": "max_iterations_reached",
            "iterations": iteration,
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
        }

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
