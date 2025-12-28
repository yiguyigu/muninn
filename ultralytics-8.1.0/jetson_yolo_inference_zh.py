"""Jetson IMX219 CSI camera + YOLOv8 realtime detection script.

已验证平台思路：Jetson Orin Nano + IMX219 + Ultralytics 8.1.0
"""

import os
import time

import cv2
from ultralytics import YOLO


def gstreamer_pipeline(
	sensor_id: int = 0,
	capture_width: int = 1920,
	capture_height: int = 1080,
	display_width: int = 640,
	display_height: int = 480,
	framerate: int = 30,
	flip_method: int = 0,
):
	"""构造 IMX219 CSI 摄像头的 GStreamer 管道字符串。"""

	# 采用 Jetson 官方示例推荐的写法，显式指定 NV12 -> BGRx -> BGR
	return (
		"nvarguscamerasrc sensor-id=%d ! "
		"video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=%d/1 ! "
		"nvvidconv flip-method=%d ! "
		"video/x-raw, width=%d, height=%d, format=BGRx ! "
		"videoconvert ! "
		"video/x-raw, format=BGR ! appsink drop=True sync=False"
		% (
			sensor_id,
			capture_width,
			capture_height,
			framerate,
			flip_method,
			display_width,
			display_height,
		)
	)


def main():
	# 检测是否有图形界面（SSH 无 DISPLAY 时不使用 imshow）
	headless = not bool(os.environ.get("DISPLAY"))
	if headless:
		print("[INFO] 检测到无 DISPLAY，使用无窗口(headless)模式运行，不调用 cv2.imshow")

	# ========== 配置参数 ==========
	model_path = "train_model/best.pt"  # 你的训练模型路径
	conf_threshold = 0.5
	iou_threshold = 0.45
	imgsz = 640

	# ========== 加载模型 ==========
	print("[INFO] 正在加载 YOLOv8 模型...")
	model = YOLO(model_path, task="detect")
	print(f"[INFO] 模型加载成功: {model_path}")
	print(f"[INFO] 类别: {model.names}")

	# ========== 打开 CSI 摄像头 ==========
	print("[INFO] 正在打开 CSI 摄像头...")
	pipeline = gstreamer_pipeline(
		sensor_id=0,
		capture_width=1920,
		capture_height=1080,
		display_width=640,
		display_height=480,
		framerate=30,
		flip_method=0,
	)
	print("[INFO] 使用的 GStreamer 管道:\n", pipeline)

	cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

	if not cap.isOpened():
		print("[ERROR] 无法打开 CSI 摄像头 (GStreamer CAP_GSTREAMER 失败)")
		print("[TIPS] 请检查:")
		print("  1. 摄像头排线是否正确连接")
		print("  2. 在系统中能否用 gst-launch-1.0 正常预览")
		return

	print("[INFO] 摄像头已就绪")
	if not headless:
		print("[INFO] 按 'q' 退出, 's' 截图保存")
	else:
		print("[INFO] Headless 模式：不弹出窗口，如需查看结果可保存图片")

	# ========== 主循环 ==========
	frame_count = 0
	start_time = time.time()
	fps = 0.0

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				print("[WARNING] 无法读取帧")
				continue

			frame_count += 1

			# YOLOv8 推理（对单帧）
			results = model.predict(
				source=frame,
				conf=conf_threshold,
				iou=iou_threshold,
				imgsz=imgsz,
				verbose=False,
				device=0,
			)

			# 提取检测框坐标和类别信息
			boxes = results[0].boxes
			xyxy = boxes.xyxy  # [N, 4]，每行为 [x1, y1, x2, y2]
			cls = boxes.cls    # [N]，类别索引
			conf = boxes.conf  # [N]，置信度

			# 示例：打印当前帧所有目标的坐标（可按需改成只取某一类）
			if len(xyxy) > 0:
				print("[DETECT] 本帧检测到目标数量:", len(xyxy))
				for i in range(len(xyxy)):
					x1, y1, x2, y2 = xyxy[i].tolist()
					cls_id = int(cls[i].item())
					label = model.names.get(cls_id, str(cls_id))
					conf_i = float(conf[i].item())
					# 这里打印左上(x1,y1)、右下(x2,y2)，以及类别和置信度
					print(f"  - {label}: conf={conf_i:.2f}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

			# 绘制检测结果
			annotated_frame = results[0].plot()

			# 计算 FPS
			elapsed_time = time.time() - start_time
			if elapsed_time > 0:
				fps = frame_count / elapsed_time

			# 显示 FPS 和检测数量
			cv2.putText(
				annotated_frame,
				f"FPS: {fps:.1f}",
				(10, 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 0),
				2,
			)

			detections = len(results[0].boxes)
			cv2.putText(
				annotated_frame,
				f"Detections: {detections}",
				(10, 60),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 0),
				2,
			)

			if not headless:
				# 显示结果（有图形界面时）
				cv2.imshow("YOLOv8 CSI Detection", annotated_frame)

				# 键盘控制
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					print("[INFO] 用户退出")
					break
				elif key == ord("s"):
					filename = f"capture_{int(time.time())}.jpg"
					cv2.imwrite(filename, annotated_frame)
					print(f"[INFO] 截图已保存: {filename}")

	except KeyboardInterrupt:
		print("\n[INFO] 检测到 Ctrl+C, 正在退出...")

	finally:
		cap.release()
		cv2.destroyAllWindows()
		print(f"[INFO] 总帧数: {frame_count}, 平均 FPS: {fps:.2f}")
		print("[INFO] 程序已退出")


if __name__ == "__main__":
	main()