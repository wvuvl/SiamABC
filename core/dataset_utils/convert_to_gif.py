import cv2

import imageio


paths = ['/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/air_show.mp4',
        #'/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/gifs/biker_comp.mp4',
        #  '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/gifs/car_comp.mp4',
        #  '/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/gifs/coke_comp.mp4'
        ]
image_lst = []
i = 0
for path in paths:
    cap = cv2.VideoCapture(path)
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame,(int(0.5*frame.shape[1]),int(0.5*frame.shape[0])))
        image_lst.append(frame_rgb)

        # cv2.imshow('a', frame)
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break
        i +=1

cap.release()
# cv2.destroyAllWindows()

# Convert to gif using the imageio.mimsave method
imageio.mimsave('/media/ramzaveri/12F9CADD61CB0337/cell_tracking/datasets/avist/air_show.gif', image_lst, fps=30)