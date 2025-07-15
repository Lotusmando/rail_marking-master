#!/usr/bin/env python
import os
import sys
import cv2
import datetime
import numpy as np

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-video_path", type=str, required=True)
    parser.add_argument("-output_video_path", type=str, default="result.MP4")
    parser.add_argument("--bottom_region", type=float, default=0.25,
                        help="画面底部区域占比，用于检测铁轨边缘")
    parser.add_argument("--min_points", type=int, default=30,
                        help="拟合直线所需的最小点数（左右通用）")
    parser.add_argument("--edge_threshold", type=int, default=100,
                        help="Canny边缘检测阈值（左右通用）")
    parser.add_argument("--show_region", type=bool, default=True,
                        help="是否显示检测区域")
    parser.add_argument("--line_length", type=int, default=150,
                        help="显示的切线长度(像素)")
    parser.add_argument("--red_threshold", type=int, default=150,
                        help="红色区域的阈值")
    parser.add_argument("--center_offset", type=int, default=100,
                        help="从中心向两侧扩展的像素距离，用于提取边缘")
    parser.add_argument("--min_rail_width", type=int, default=50,
                        help="左右铁轨之间的最小宽度(像素)")
    parser.add_argument("--history_size", type=int, default=6,  # 减小历史窗口
                        help="历史平滑的帧数")
    parser.add_argument("--straight_threshold", type=float, default=0.6,
                        help="判定为直线轨道的阈值（0-1）")
    parser.add_argument("--hough_threshold", type=int, default=60,
                        help="霍夫变换的阈值")
    parser.add_argument("--hough_min_length", type=int, default=80,
                        help="霍夫变换检测的最小直线长度")
    parser.add_argument("--hough_max_gap", type=int, default=15,
                        help="霍夫变换检测的最大直线间隙")
    parser.add_argument("--curvature_threshold", type=float, default=0.15,  # 降低曲率阈值
                        help="区分直道和弯道的曲率阈值")
    parser.add_argument("--direction_threshold", type=float, default=15.0,
                        help="判断弯道方向的距离差阈值(像素)")
    parser.add_argument("--offset_threshold", type=float, default=30.0,
                        help="判定为直道的最大偏移量(像素)")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="判定为直道的最小置信度(0-1)")

    args = parser.parse_args()
    return args


def calculate_curvature(left_line, right_line, frame_center_x, args):
    """计算轨道曲率和方向（左弯/右弯）"""
    if left_line is None or right_line is None:
        return 0.0, "Unknown"
    
    # 提取左右铁轨参数
    left_vx, left_vy, left_x, left_y = left_line
    right_vx, right_vy, right_x, right_y = right_line
    
    # 计算左右铁轨的斜率
    left_slope = left_vy / left_vx if left_vx != 0 else float('inf')
    right_slope = right_vy / right_vx if right_vx != 0 else float('inf')
    
    # 基于斜率的曲率计算
    slope_curvature = abs(left_slope - right_slope)
    
    # 基于位置的距离差计算
    left_distance = abs(left_x - frame_center_x)
    right_distance = abs(right_x - frame_center_x)
    position_diff = right_distance - left_distance  # 右大：右弯，左大：左弯
    
    # 综合曲率
    curvature = slope_curvature * 0.7 + abs(position_diff) * 0.01
    
    # 判断方向
    direction = "Unknown"
    if position_diff > args.direction_threshold:
        direction = "Right Curve"
    elif position_diff < -args.direction_threshold:
        direction = "Left Curve"
    else:
        if slope_curvature > args.curvature_threshold * 1.5:
            direction = "Right Curve" if left_slope > right_slope else "Left Curve"
        else:
            direction = "Straight"
    
    return curvature, direction


def get_rail_tangents_from_overlay(overlay, args, frame_center_x):
    height, width = overlay.shape[:2]
    
    # 只关注画面底部的区域
    bottom_region = int(height * (1 - args.bottom_region))
    bottom_overlay = overlay[bottom_region:, :, :].copy()
    
    # 提取红色区域（左右通用阈值）
    lower_red = np.array([0, 0, args.red_threshold])
    upper_red = np.array([100, 100, 255])
    red_mask = cv2.inRange(bottom_overlay, lower_red, upper_red)
    
    # 增强边缘检测（减少噪声影响）
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, bottom_region, 0.0, "Unknown"
    
    # 提取轮廓点并转换回原图坐标
    all_points = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue
        for point in contour:
            x, y = point[0]
            all_points.append((x, y + bottom_region))
    
    if len(all_points) < args.min_points:
        return None, None, None, bottom_region, 0.0, "Unknown"
    
    # 转换为numpy数组
    all_points = np.array(all_points)
    
    # 计算铁轨区域的实际中线（黄色线）
    rail_center_x = np.mean(all_points[:, 0])
    
    # 分割左右铁轨边缘点
    left_points = all_points[all_points[:, 0] < rail_center_x]
    right_points = all_points[all_points[:, 0] >= rail_center_x]
    
    # 进一步筛选靠近中线的点
    left_points = left_points[left_points[:, 0] > (rail_center_x - args.center_offset)]
    right_points = right_points[right_points[:, 0] < (rail_center_x + args.center_offset)]
    
    # 检查左右铁轨之间的宽度
    if len(left_points) > 0 and len(right_points) > 0:
        left_center = np.mean(left_points[:, 0])
        right_center = np.mean(right_points[:, 0])
        rail_width = right_center - left_center
        
        if rail_width < args.min_rail_width:
            if len(left_points) < len(right_points):
                left_points = np.array([])
            else:
                right_points = np.array([])
    
    # 拟合左右铁轨的直线
    left_line = None
    right_line = None
    
    if len(left_points) >= args.min_points:
        [vx, vy, x, y] = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
        left_line = (vx, vy, x, y)
    
    if len(right_points) >= args.min_points:
        [vx, vy, x, y] = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)
        right_line = (vx, vy, x, y)
    
    # 计算曲率和方向
    curvature, direction = calculate_curvature(left_line, right_line, frame_center_x, args)
    
    return left_line, right_line, rail_center_x, bottom_region, curvature, direction


def main():
    args = get_args()
    segmentation_handler = RailtrackSegmentationHandler(args.snapshot, BiSeNetV2Config())

    capture = cv2.VideoCapture(args.video_path)
    if not capture.isOpened():
        raise Exception("failed to open {}".format(args.video_path))

    width = int(capture.get(3))
    height = int(capture.get(4))
    frame_center_x = width // 2  # 蓝色画面中线（基准线）

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        print("警告: 无法获取原视频帧率，使用默认值 30.0fps")
    out_video = cv2.VideoWriter(args.output_video_path, fourcc, fps, (width, height))

    _total_ms = 0
    count_frame = 0
    
    # 历史记录用于平滑
    left_history = []
    right_history = []
    rail_center_history = []  # 黄色铁轨中线历史
    offset_history = []       # 偏移量历史（蓝色线与黄色线的差值）
    direction_history = []    # 方向历史

    while capture.isOpened():
        ret, frame = capture.read()
        count_frame += 1
        if not ret:
            break

        start = datetime.datetime.now()
        mask, overlay = segmentation_handler.run(frame, only_mask=False)
        _total_ms += (datetime.datetime.now() - start).total_seconds() * 1000

        # 获取铁轨信息
        left_line, right_line, raw_center_x, bottom_region, current_curvature, current_direction = get_rail_tangents_from_overlay(overlay, args, frame_center_x)
        
        # 1. 优化铁轨中线（黄色线）：滑动平均减少抖动
        if raw_center_x is not None:
            rail_center_history.append(raw_center_x)
            if len(rail_center_history) > args.history_size:
                rail_center_history.pop(0)
            # 加权平均，最近帧权重更高
            weights = np.linspace(0.5, 1.0, len(rail_center_history))
            weights /= weights.sum()
            smoothed_center_x = np.average(rail_center_history, weights=weights)
        else:
            smoothed_center_x = np.mean(rail_center_history) if rail_center_history else frame_center_x

        # 2. 计算偏移量：蓝色画面中线与黄色铁轨中线的差值
        offset_pixels = smoothed_center_x - frame_center_x  # 正数：铁轨偏右，负数：铁轨偏左
        offset_history.append(offset_pixels)
        if len(offset_history) > args.history_size:
            offset_history.pop(0)
        
        # 平滑后的偏移量
        smoothed_offset = np.mean(offset_history)
        offset_text = f"Offset: {smoothed_offset:.1f}px"

        # 3. 平滑方向判断
        direction_history.append(current_direction)
        if len(direction_history) > args.history_size:
            direction_history.pop(0)
        
        # 统计各方向出现次数
        direction_counts = {
            "Left Curve": direction_history.count("Left Curve"),
            "Right Curve": direction_history.count("Right Curve"),
            "Straight": direction_history.count("Straight"),
            "Unknown": direction_history.count("Unknown")
        }
        
        # 计算直道置信度
        straight_ratio = direction_counts["Straight"] / len(direction_history) if direction_history else 0.0
        
        # 选择出现次数最多的方向作为最终方向
        most_common_direction = max(direction_counts, key=direction_counts.get)
        # 简化方向显示（只保留左右弯和直道）
        display_direction = most_common_direction if most_common_direction != "Unknown" else "Straight"

        # 4. 改进的直道判定逻辑
        is_straight = (abs(smoothed_offset) < args.offset_threshold) and (straight_ratio > args.min_confidence)

        # 覆盖方向判断（如果直道条件满足）
        if is_straight:
            display_direction = "Straight"

        # 更新左右铁轨历史
        if left_line is not None:
            left_history.append(left_line)
            if len(left_history) > args.history_size:
                left_history.pop(0)
        if right_line is not None:
            right_history.append(right_line)
            if len(right_history) > args.history_size:
                right_history.pop(0)

        # 计算左右铁轨历史平均值
        avg_left_line = None
        if len(left_history) > 0:
            avg_vx = np.mean([h[0] for h in left_history])
            avg_vy = np.mean([h[1] for h in left_history])
            avg_x = np.mean([h[2] for h in left_history])
            avg_y = np.mean([h[3] for h in left_history])
            avg_left_line = (avg_vx, avg_vy, avg_x, avg_y)
        
        avg_right_line = None
        if len(right_history) > 0:
            avg_vx = np.mean([h[0] for h in right_history])
            avg_vy = np.mean([h[1] for h in right_history])
            avg_x = np.mean([h[2] for h in right_history])
            avg_y = np.mean([h[3] for h in right_history])
            avg_right_line = (avg_vx, avg_vy, avg_x, avg_y)

        # 绘制基准线和铁轨中线
        # 蓝色画面中线（基准线）
        cv2.line(overlay, (frame_center_x, 0), (frame_center_x, height), (255, 0, 0), 2)
        
        # 黄色铁轨中线
        smoothed_center_x_int = int(smoothed_center_x)
        cv2.line(overlay, (smoothed_center_x_int, 0), (smoothed_center_x_int, height), (0, 255, 255), 2)
        
        # 显示检测区域（可选）
        if args.show_region and bottom_region is not None:
            cv2.rectangle(overlay, (0, bottom_region), (width, height), (0, 255, 0), 1)

        # 左上角显示方向和偏移量信息
        # 1. 方向信息
        dir_color = (0, 255, 0) if display_direction == "Straight" else (0, 165, 255)
        cv2.putText(overlay, display_direction, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)
        
        # 2. 偏移量信息（蓝色线与黄色线的差值）
        offset_color = (0, 255, 0) if abs(smoothed_offset) < args.offset_threshold else (0, 0, 255)
        cv2.putText(overlay, offset_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, offset_color, 2)

        # 绘制铁轨切线
        left_drawn = False
        right_drawn = False
        
        # 绘制左铁轨
        if avg_left_line is not None:
            avg_vx, avg_vy, avg_x, avg_y = avg_left_line
            if smoothed_center_x is None or avg_x < smoothed_center_x:
                start_x = int(avg_x - avg_vx * args.line_length)
                start_y = int(avg_y - avg_vy * args.line_length)
                end_x = int(avg_x + avg_vx * args.line_length)
                end_y = int(avg_y + avg_vy * args.line_length)
                cv2.line(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
                left_drawn = True
        
        # 绘制右铁轨
        if avg_right_line is not None:
            avg_vx, avg_vy, avg_x, avg_y = avg_right_line
            # 修复：添加完整的条件判断
            if smoothed_center_x is None or avg_x >= smoothed_center_x:
                start_x = int(avg_x - avg_vx * args.line_length)
                start_y = int(avg_y - avg_vy * args.line_length)
                end_x = int(avg_x + avg_vx * args.line_length)
                end_y = int(avg_y + avg_vy * args.line_length)
                cv2.line(overlay, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                right_drawn = True

        cv2.imshow("result", overlay)
        out_video.write(overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"processing time per frame: {_total_ms / count_frame:.2f}[ms]")

    capture.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
