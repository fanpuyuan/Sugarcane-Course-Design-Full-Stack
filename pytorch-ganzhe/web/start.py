# 文件名: start.py
import subprocess
import sys
import os


def main():
    print("正在启动 Streamlit 应用...")
    try:
        # streamlit.exe 在 python.exe 同级的 Scripts 目录下
        python_dir = os.path.dirname(sys.executable)
        streamlit_exe = os.path.join(python_dir, "Scripts", "streamlit.exe")

        # 如果 Scripts 子目录里没有，尝试同级目录（兼容不同 conda 版本）
        if not os.path.exists(streamlit_exe):
            streamlit_exe = os.path.join(python_dir, "streamlit.exe")

        command = [streamlit_exe, "run", "test.py"]
        print(f"使用 streamlit 路径: {streamlit_exe}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit 应用启动失败，退出码: {e.returncode}")
        print(f"错误信息: {e}")
    except FileNotFoundError:
        print("错误: 找不到 'streamlit' 命令。请确保已安装 Streamlit 并且它在 PATH 中。")
        print("尝试运行: pip install streamlit")
    except Exception as e:
        print(f"启动过程中发生未知错误: {e}")


if __name__ == "__main__":
    main()