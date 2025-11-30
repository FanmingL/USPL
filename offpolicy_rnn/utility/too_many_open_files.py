import os

def list_open_files(logger_func):
    pid = os.getpid()
    fd_dir = f"/proc/{pid}/fd"
    try:
        files = os.listdir(fd_dir)
        open_files = []
        for fd in files:
            try:
                file_path = os.readlink(os.path.join(fd_dir, fd))
                open_files.append((fd, file_path))
            except OSError:
                pass
        return open_files
    except FileNotFoundError:
        logger_func("Cannot access /proc. Are you on a non-UNIX system?")
        return []

def close_open_pipes(logger_func):
    open_files = list_open_files(logger_func)  # 或使用 list_open_files_with_psutil()
    for fd, file_path in open_files:
        if file_path.startswith('pipe:'):
            try:
                os.close(int(fd))
                logger_func(f"Closed pipe: {file_path}")
            except OSError as e:
                logger_func(f"Failed to close pipe {file_path}: {e}")

