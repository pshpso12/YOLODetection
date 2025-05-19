import cv2
import numpy as np
import time

scale = 0.00392
classes = None
with open('./loadmodel/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet("./loadmodel/yolov113_last.weights", "./loadmodel/yolov113.cfg")

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def network(image):

    Width = image.shape[1]
    Height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image, scale, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))
    class_ids = []
    sumaaa = 0
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                if class_id == 0:
                    sumaaa +=1
                elif class_id  == 1:
                    sumaaa -=1
                

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print("Cars- Humans : ", sumaaa)
        if sumaaa > 5 :
            print('Too much car')
        elif sumaaa < -5 :
            print('Too much person')
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h),classes,COLORS)
    return image

output_image_width = int(1280/2)
output_image_height=int(720/2)
capture = cv2.VideoCapture('./img_video/videoplayback.mp4')
cnt = 0
while True:
    ret, image = capture.read()
    image = cv2.resize(image,(output_image_width,output_image_height),interpolation=cv2.INTER_AREA)
    cnt += 1
    if ret and cnt % 2 == 0:

        height = image.shape[0]
        width = image.shape[1]

        #image = image[0:int(height / 1.5), 0: width]
        start = time.time()
        image = network(image)
        end = time.time()
        #print(end-start)
        cv2.imshow('res', image)

        if cv2.waitKey(10) & 0xff == ord('q'):
            break