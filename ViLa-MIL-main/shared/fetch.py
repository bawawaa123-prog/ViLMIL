import paramiko
import argparse
import os
import stat

def download_from_server(hostname, port, username, password, remote_path, local_path):
    """
    通过 SFTP 从远程服务器下载文件或整个目录。
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        print(f"正在连接到 {hostname}...")
        ssh.connect(hostname=hostname, port=port, username=username, password=password)
        sftp = ssh.open_sftp()
        
        file_attr = sftp.stat(remote_path)
        
        if stat.S_ISDIR(file_attr.st_mode):
            print(f"[{remote_path}] 是一个目录。")
            if not os.path.exists(local_path):
                os.makedirs(local_path)
                
            for filename in sftp.listdir(remote_path):
                remote_filepath = remote_path + '/' + filename
                local_filepath = os.path.join(local_path, filename)
                print(f"正在下载文件: {filename} -> {local_filepath}")
                sftp.get(remote_filepath, local_filepath)
        else:
            # 如果是单个文件，判断 local_path 是不是一个文件夹
            if os.path.isdir(local_path):
                # 提取远程文件的名字 (比如从 /path/to/test.txt 提取出 test.txt)
                filename = os.path.basename(remote_path)
                # 拼接成完整的本地文件路径
                final_local_path = os.path.join(local_path, filename)
            else:
                # 如果用户已经指定了具体文件名，就直接用
                final_local_path = local_path
                
            print(f"正在下载文件: {remote_path} -> {final_local_path}")
            sftp.get(remote_path, final_local_path)
            
        print("传输完成！")
        sftp.close()

    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 Linux 服务器下载文件或文件夹的工具")
    
    # 将 required=True 去掉，替换为 default="..."
    parser.add_argument("--host", default="172.23.206.200", help="服务器 IP 地址 (默认: 172.23.206.200)")
    parser.add_argument("--port", type=int, default=1025, help="SSH 端口 (默认: 22)")
    parser.add_argument("--user", default="ljh", help="用户名 (默认: ljh)")
    parser.add_argument("--password", default="ljh136807", help="密码")
    
    # 目标路径依然需要用户自己输入
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
