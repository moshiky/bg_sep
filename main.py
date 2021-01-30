
import cv2
import time


def main():
    # create capture object
    cap = cv2.VideoCapture(0)
    print('initial res:', cap.get(3), cap.get(4))

    # change to 1920 X 1080
    print('set new res: 1920 X 1080')
    cap.set(3, 1920)
    cap.set(4, 1080)

    print('new res:', cap.get(3), cap.get(4))

    start_time = time.time()
    wait_for = 5
    frame_counter = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame_counter += 1

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_passed = time.time() - start_time
        if time_passed > wait_for and int(time_passed) % 5 == 0:
            print('curr fps:', frame_counter / time_passed)
            wait_for += 5

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
