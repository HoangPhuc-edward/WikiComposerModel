import os
import subprocess

# 1. Cấu hình đường dẫn thư mục chứa video
video_dir = 'datasets/video/mp4'

# 2. Danh sách các định dạng cần chuyển đổi
target_extensions = ('.webm', '.mkv')

def convert_to_mp4(input_path, output_path):
    # Lệnh ffmpeg: 
    # -i: file đầu vào
    # -c:v copy: giữ nguyên codec video (không nén lại để chạy cực nhanh)
    # -c:a aac: chuyển audio sang định dạng aac (tương thích tốt nhất với mp4)
    # -y: ghi đè nếu file đã tồn tại
    command = [
        'ffmpeg', '-i', input_path,
        '-c:v', 'copy', 
        '-c:a', 'aac', 
        '-strict', 'experimental',
        '-y', output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        return False

print("🚀 Bắt đầu quá trình chuẩn hóa video sang MP4...")

files = os.listdir(video_dir)
count = 0

for filename in files:
    if filename.lower().endswith(target_extensions):
        input_file = os.path.join(video_dir, filename)
        # Tạo tên file mới: thay đuôi cũ bằng .mp4
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(video_dir, f"{base_name}.mp4")
        
        print(f"🔄 Đang chuyển đổi: {filename} -> {base_name}.mp4")
        
        if convert_to_mp4(input_file, output_file):
            # Sau khi chuyển đổi thành công, xóa file gốc để tiết kiệm bộ nhớ
            os.remove(input_file)
            count += 1
            print(f"✅ Thành công!")
        else:
            print(f"❌ Lỗi khi xử lý file: {filename}")

print(f"\n✨ Hoàn thành! Đã chuyển đổi thành công {count} file.")