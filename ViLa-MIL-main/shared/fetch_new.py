import paramiko
import argparse
import os
import stat

def download_recursive(sftp, remote_dir, local_dir):
    """
    核心递归函数：遇到文件就下载，遇到文件夹就钻进去继续找
    """
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    for filename in sftp.listdir(remote_dir):
        # 拼接远程和本地的完整路径
        remote_filepath = remote_dir.rstrip('/') + '/' + filename
        local_filepath = os.path.join(local_dir, filename)
        
        try:
            # 获取当前项目的属性
            file_attr = sftp.stat(remote_filepath)
            
            # 判断是文件夹还是文件
            if stat.S_ISDIR(file_attr.st_mode):
                print(f"📂 发现子文件夹: [{filename}]，正在钻入...")
                # 如果是文件夹，调用自己（递归）
                download_recursive(sftp, remote_filepath, local_filepath)
            else:
                # 如果是文件，正常下载
                print(f"📄 正在下载文件: {filename} -> {local_filepath}")
                sftp.get(remote_filepath, local_filepath)
        except Exception as e:
            print(f"❌ 处理 {remote_filepath} 时出错: {e}")


def download_from_server(hostname, port, username, password, remote_path, local_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"🔌 正在连接到 {hostname}:{port}...")
        ssh.connect(hostname=hostname, port=port, username=username, password=password)
        sftp = ssh.open_sftp()
        
        file_attr = sftp.stat(remote_path)
        
        if stat.S_ISDIR(file_attr.st_mode):
            print(f"📦 目标 [{remote_path}] 是一个目录，开始递归打包下载...")
            
            # 贴心小优化：如果你没指定本地存成什么名字，就用远程文件夹的名字建一个
            if local_path == './' or local_path == '.':
                 folder_name = os.path.basename(remote_path.rstrip('/'))
                 local_path = os.path.join(local_path, folder_name)
                 
            download_recursive(sftp, remote_path, local_path)
        else:
            # 单文件的下载逻辑（保留了上一次的修复）
            if os.path.isdir(local_path):
                filename = os.path.basename(remote_path)
                final_local_path = os.path.join(local_path, filename)
            else:
                final_local_path = local_path
                
            print(f"📄 正在下载文件: {remote_path} -> {final_local_path}")
            sftp.get(remote_path, final_local_path)
            
        print("✅ 全部传输完成！")
        sftp.close()

    except Exception as e:
        print(f"💥 发生致命错误: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Linux 服务器下载文件或文件夹的工具")
    
    # 你的默认参数都在这里 (注意端口已经是 1025 了)
    parser.add_argument("--host", default="172.23.206.200", help="服务器 IP 地址")
    parser.add_argument("--port", type=int, default=1025, help="SSH 端口")
    parser.add_argument("--user", default="ljh", help="用户名")
    parser.add_argument("--password", default="ljh136807", help="密码")
    
    parser.add_argument("--target", required=True, help="要下载的远程文件或文件夹的绝对路径")
    parser.add_argument("--dest", default="./", help="保存到本地的路径 (默认: 当前目录)")

    args = parser.parse_args()

    download_from_server(
        hostname=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        remote_path=args.target,
        local_path=args.dest
    )
