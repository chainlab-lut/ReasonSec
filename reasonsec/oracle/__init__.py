from reasonsec.config import Config, ConfigError
from reasonsec.oracle.base import CachedOracle, CompositeOracle, SecurityOracle
from reasonsec.oracle.codeql_oracle import CodeQLOracle
from reasonsec.oracle.regex_oracle import RegexOracle
from reasonsec.oracle.semgrep_oracle import SemgrepOracle
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)

_ORACLE_TYPES = {
    "semgrep": SemgrepOracle,
    "regex": RegexOracle,
    "codeql": CodeQLOracle,
}


def build_oracle(config: Config, name: str | None = None) -> SecurityOracle:
    oracle_name = name or config.require_str("oracle.primary")
    definition = config.require_section(f"oracle.definitions.{oracle_name}")
    component_names = [str(item) for item in definition.require_list("components")]
    components: list[SecurityOracle] = []
    for component in component_names:
        if component not in _ORACLE_TYPES:
            raise ConfigError(f"unknown oracle component '{component}'; available components are {sorted(_ORACLE_TYPES)}")
        components.append(_ORACLE_TYPES[component](config))
    selection_strategy = config.require_str("oracle.cwe_selection_strategy")
    oracle: SecurityOracle
    if len(components) == 1:
        oracle = components[0]
        oracle._selection_strategy = selection_strategy
        oracle.name = oracle_name
    else:
        oracle = CompositeOracle(components, name=oracle_name, selection_strategy=selection_strategy)
    if config.require_bool("oracle.cache.enabled"):
        cache_path = config.require_path("oracle.cache.path", must_exist=False)
        oracle = CachedOracle(oracle, cache_path)
    LOGGER.info("constructed oracle '%s' from components %s", oracle_name, component_names)
    return oracle


__all__ = [
    "build_oracle",
    "CachedOracle",
    "CodeQLOracle",
    "CompositeOracle",
    "RegexOracle",
    "SecurityOracle",
    "SemgrepOracle",
]
