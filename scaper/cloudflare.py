import subprocess

def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

# xem trạng thái đăng ký
print(run(["warp-cli", "registration", "show"]))

