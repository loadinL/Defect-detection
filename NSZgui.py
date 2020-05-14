from PIL import Image,ImageTk
from tkinter.filedialog import askopenfilename
import time
import tkinter as tk  # 使用Tkinter前需要先导
import os
import io
import argparse
from sys import platform
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

##对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
def resize( w_box, h_box, pil_image): #参数是：要适应的窗口宽、高、Image.open后的图片
    w, h = pil_image.size #获取图像的原始大小
    f1 = 1.0*w_box/w
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


# 第1步，实例化object，建立窗口window
window = tk.Tk()

# 第2步，给窗口的可视化起名字
window.title('尿试纸检测')

# 第3步，设定窗口的大小(长 * 宽)
window.geometry('600x500')  # 这里的乘是小x

a=tk.Label(window, text='图像显示', bg='white', fg='black', font=('Arial', 12), width=30, height=2)
a.pack(side='top',fill='x')


w_box = 800   #期望图像显示的大小（窗口大小）
h_box = 800

def choosepic():
    global path_
    path_ = askopenfilename()
    path.set(path_)
    img_open = Image.open(file_entry.get())
    img_open__resized = resize( w_box, h_box, img_open)
    img = ImageTk.PhotoImage(img_open__resized)
    l.config(image=img)
    l.image = img  # keep a reference

path = tk.StringVar()

file_entry = tk.Entry(window, state='readonly', text=path)


l = tk.Label(window, bg='Gainsboro', fg='white', font=('Arial', 12),width=30, height=2)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
l.pack(side='top', fill='both', expand='yes',padx='5', pady='5')

b = tk.Button(window, text='选择图片', font=('Arial', 12), width=10, height=1, command=choosepic)
b.pack(side='left', fill='none', expand='yes', padx='300', pady='50')

var = tk.StringVar()
d = tk.Label(window, textvariable=var, bg='white', fg='black', font=('Arial', 12), width=60, height=2)
# 说明： bg为背景，fg为字体颜色，font为字体，width为长，height为高，这里的长和高是字符的长和高，比如height=2,就是标签有2个字符这么高
d.pack(side='left', fill='both', expand='yes')

def DETECT():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/utp.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/best_250.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default=path_, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')  # 0.3
    parser.add_argument('--nms-thres', type=float, default=0.1, help='iou threshold for non-maximum suppression')  # 0.5
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    save_txt = False
    save_img = False
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            T = time.time() - t
            print('%sDone. (%.3fs)' % (s, T))

            var.set('%sDone. (%.3fs)' % (s, T))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
        #detect()
    img_open = Image.open(save_path)
    img_open__resized = resize(w_box, h_box, img_open)
    img = ImageTk.PhotoImage(img_open__resized)
    l.config(image=img)
    l.image = img




    # im = Image.open(save_path)
    # im.show()

c = tk.Button(window, text='开始检测', font=('Arial', 12), width=10, height=1, command=DETECT)
c.pack(side='right', fill='none', expand='yes', padx='300', pady='50')


# 第6步，主窗口循环显示
window.mainloop()
