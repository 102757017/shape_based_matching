import cv2
import numpy as np
import os
import shape_based_matching_py

def angle_test(mode, use_rot):
    # 使用相对路径，确保跨平台兼容性
    base_dir = os.path.join("test", "case1")
    
    # 确保目录存在
    os.makedirs(base_dir, exist_ok=True)
    
    detector = shape_based_matching_py.Detector(128, [4, 8])
    
    if mode != 'test':
        # 使用跨平台路径连接
        train_img_path = os.path.join(base_dir, "train.png")
        img = cv2.imread(train_img_path)
        
        if img is None:
            print(f"错误: 无法读取训练图像 {train_img_path}")
            return
        
        # 提取ROI区域
        img = img[110:380, 130:400]
        mask = np.ones((img.shape[0], img.shape[1]), np.uint8)
        mask *= 255

        padding = 100
        padded_img = np.zeros((img.shape[0] + 2 * padding, 
                              img.shape[1] + 2 * padding, 
                              img.shape[2]), np.uint8)
        padded_mask = np.zeros((padded_img.shape[0], padded_img.shape[1]), np.uint8)

        padded_img[padding:padded_img.shape[0]-padding, 
                  padding:padded_img.shape[1]-padding, :] = img[:, :, :]
        padded_mask[padding:padded_img.shape[0]-padding, 
                   padding:padded_img.shape[1]-padding] = mask[:, :]

        # 创建形状信息生产者
        shapes = shape_based_matching_py.shapeInfo_producer(padded_img, padded_mask)
        shapes.angle_range = [0, 360]
        shapes.angle_step = 1
        shapes.scale_range = [1]
        shapes.produce_infos()
        
        infos_have_templ = []
        class_id = "test"
        is_first = True
        first_id = 0
        first_angle = 0

        for info in shapes.infos:
            to_show = shapes.src_of(info)

            templ_id = 0
            if is_first:
                templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))
                first_id = templ_id
                first_angle = info.angle

                if use_rot:
                    is_first = False
            else:
                templ_id = detector.addTemplate_rotate(class_id, first_id,
                                                       info.angle - first_angle,
                    shape_based_matching_py.CV_Point2f(padded_img.shape[1] / 2.0, 
                                                      padded_img.shape[0] / 2.0))
            
            # 检查模板是否添加成功
            if templ_id != -1:
                templ = detector.getTemplates(class_id, templ_id)
                if templ:  # 确保获取到模板
                    for feat in templ[0].features:
                        to_show = cv2.circle(to_show, 
                                            (feat.x + templ[0].tl_x, feat.y + templ[0].tl_y), 
                                            3, (0, 0, 255), -1)
                cv2.imshow("show templ", to_show)
                cv2.waitKey(1)
                infos_have_templ.append(info)

        # 使用跨平台路径保存
        templ_yaml_path = os.path.join(base_dir, "%s_templ.yaml")
        detector.writeClasses(templ_yaml_path)
        
        info_yaml_path = os.path.join(base_dir, "test_info.yaml")
        shapes.save_infos(infos_have_templ, info_yaml_path)
        
        print(f"模板训练完成，保存到 {base_dir}")
        
    else:
        # 测试模式
        ids = ['test']
        
        # 使用跨平台路径加载模板
        templ_yaml_path = os.path.join(base_dir, "%s_templ.yaml")
        detector.readClasses(ids, templ_yaml_path)

        producer = shape_based_matching_py.shapeInfo_producer()
        
        # 使用跨平台路径加载信息
        info_yaml_path = os.path.join(base_dir, "test_info.yaml")
        infos = producer.load_infos(info_yaml_path)
        
        # 使用跨平台路径读取测试图像
        test_img_path = os.path.join(base_dir, "test.png")
        test_img = cv2.imread(test_img_path)
        
        if test_img is None:
            print(f"错误: 无法读取测试图像 {test_img_path}")
            return

        padding = 250
        padded_img = np.zeros((test_img.shape[0] + 2 * padding, 
                              test_img.shape[1] + 2 * padding, 
                              test_img.shape[2]), np.uint8)
        padded_img[padding:padded_img.shape[0]-padding, 
                  padding:padded_img.shape[1]-padding, :] = test_img[:, :, :]

        # 确保图像尺寸是16的倍数
        stride = 16
        img_rows = (padded_img.shape[0] // stride) * stride
        img_cols = (padded_img.shape[1] // stride) * stride
        
        # 创建符合要求的图像
        if padded_img.shape[0] != img_rows or padded_img.shape[1] != img_cols:
            img = np.zeros((img_rows, img_cols, padded_img.shape[2]), np.uint8)
            img[:, :, :] = padded_img[0:img_rows, 0:img_cols, :]
        else:
            img = padded_img
            
        # 执行匹配
        matches = detector.match(img, 90, ids)
        
        # 显示前5个匹配结果
        top5 = min(5, len(matches))
        for i in range(top5):
            match = matches[i]
            templ = detector.getTemplates("test", match.template_id)
            
            if templ:  # 确保获取到模板
                for feat in templ[0].features:
                    img = cv2.circle(img, 
                                    (feat.x + match.x, feat.y + match.y), 
                                    3, (0, 0, 255), -1)

            print(f'匹配结果 {i+1}:')
            print(f'  模板ID: {match.template_id}')
            print(f'  相似度: {match.similarity}')
            print(f'  位置: ({match.x}, {match.y})')
            
        cv2.imshow("匹配结果", img)
        cv2.waitKey(0)

def main():
    """
    主函数，提供命令行交互界面
    """
    print("=== 形状匹配演示程序 ===")
    print("1. 训练模式 (创建模板)")
    print("2. 测试模式 (使用模板匹配)")
    
    choice = input("请选择模式 (1或2): ").strip()
    
    if choice == "1":
        print("使用旋转模板优化? (y/n): ")
        use_rot_choice = input().strip().lower()
        use_rot = use_rot_choice == 'y'
        angle_test('train', use_rot)
    elif choice == "2":
        angle_test('test', True)
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
