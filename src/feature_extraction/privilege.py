from typing import Dict, List


class PrivilegeTracker:
    DEFAULT_MAPPINGS = {
        "list_files": 1,
        "read_file": 1,
        "search_web": 1,
        "list_channels": 1,
        "read_slack": 1,
        "summarize_document": 1,
        "parse_json": 1,
        "get_weather": 1,
        "get_user_info": 1,
        "read_database": 2,
        "query_database": 2,
        "write_file": 2,
        "transform_data": 2,
        "export_csv": 2,
        "http_get": 2,
        "send_slack": 2,
        "send_email": 3,
        "http_post": 3,
        "call_api": 3,
        "execute_script": 3,
        "install_package": 3,
        "delete_user": 4,
        "change_permissions": 4,
        "change_password": 4,
        "transfer_funds": 4,
        "root_shell": 5,
    }

    def __init__(self, custom_mappings: Dict[str, int] | None = None, w_priv: float = 1.5):
        self.privilege_map = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            self.privilege_map.update(custom_mappings)
        self.w_priv = w_priv

    def get_level(self, tool_name: str) -> int:
        return self.privilege_map.get(tool_name, 1)

    def calculate_dynamic_escalation(self, tool_sequence: List[str]) -> List[float]:
        if not tool_sequence:
            return []
        escalation_scores = [0.0]
        for i in range(1, len(tool_sequence)):
            prev_level = self.get_level(tool_sequence[i - 1])
            curr_level = self.get_level(tool_sequence[i])
            delta_p = curr_level - prev_level
            s_priv = max(0, delta_p) * self.w_priv
            escalation_scores.append(min(s_priv / 6.0, 1.0))
        return escalation_scores

    def calculate_cumulative_exposure(self, tool_sequence: List[str]) -> float:
        if not tool_sequence:
            return 0.0
        exposure = sum(self.get_level(tool) for tool in tool_sequence)
        return min(exposure / (5.0 * len(tool_sequence)), 1.0)
