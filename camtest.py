import cv2

cmaera = cv2.Videocmaera(
    "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4")
cmaera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cmaera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = cmaera.read()
    cv2.imshow("VideoFrame", frame)

cmaera.release()
cv2.destroyAllWindows()
