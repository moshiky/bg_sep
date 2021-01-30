
import cv2

from pixel_manager import PixelManager


def main():
    # create capture object
    cap = cv2.VideoCapture(0)

    curr_h, curr_w = cap.get(4), cap.get(3)
    print('curr res (HxW):', curr_h, 'X', curr_w)

    # # change to 1080 X 1920
    # print('set new res (HxW): 1080 X 1920')
    #
    # cap.set(4, 1080)
    # cap.set(3, 1920)
    #
    # curr_h, curr_w = cap.get(4), cap.get(3)
    # print('new res (HxW):', curr_h, 'X', curr_w)

    # init pixel manager
    px_man = PixelManager(height=curr_h, width=curr_w, sample_size=300)
    curr_stage = 0

    # start statistics collection
    print('[0] collect background statistics')
    while curr_stage == 0:

        # capture frame-by-frame
        ret, frame = cap.read()

        # update pixel statistics
        if not px_man.add_frame(frame):
            curr_stage += 1

        # test for exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('calculate statistics')
    px_man.calculate_stats()
    cv2.imshow('centroid', px_man.get_centroid())

    # start statistics collection
    print('[1] classify pixels')
    while curr_stage == 1:

        # capture frame-by-frame
        ret, frame = cap.read()

        # show original frame
        cv2.imshow('org_frame', frame)

        # get pixel mask
        mask = px_man.get_mask(frame)

        # show mask
        cv2.imshow('mask', mask)

        # test for exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
