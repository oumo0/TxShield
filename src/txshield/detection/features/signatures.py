SIG_MAP = {
    "0x82be0b96": "collateralToken", "0x23b872dd": "transferFrom",
    "0xa9059cbb": "transfer", "0x095ea7b3": "approve",
    "0x40c10f19": "mint", "0x42966c68": "burn",
    "0x4e71d92d": "withdraw", "0x2e1a7d4d": "withdraw",
    "0x3ccfd60b": "flashLoan", "0x5cffe9de": "flashLoan",
    "0x1e2e5690": "swap", "0x38ed1739": "swap",
    "0x7c025200": "flashLoan", "0xcdc2e466": "withdraw",
    "0xd0e30db0": "deposit", "0x6a627842": "mint",
    "0x313ce567": "decimals", "0x06fdde03": "name",
    "0x95d89b41": "symbol", "0x18160ddd": "totalSupply",
}

def safe_parse_hex(value):
    if not isinstance(value, str):
        return str(value)
    if value.startswith("0x") and len(value) >= 10:
        return SIG_MAP.get(value[:10].lower(), value)
    return value
