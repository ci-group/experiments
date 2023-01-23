def parse_runs_arg(cliarg: str) -> None:
    if cliarg.isnumeric():
        return [int(cliarg)]
    else:
        parts = cliarg.split(":")
        if len(parts) != 2 or not parts[0].isnumeric() or not parts[1].isnumeric():
            raise ValueError()
        low = int(parts[0])
        high = int(parts[1])
        if low > high:
            raise ValueError()
        return [i for i in range(low, high + 1)]
