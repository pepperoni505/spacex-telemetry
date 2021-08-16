import streamlink
import pytesseract
import cv2
from PIL import Image
import numpy as np

DATA_FPS = 5 # number of frames per second we should get the data at

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def getStreamURL(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError('No streams are available')

def getStreamData(url):
    stream_url = getStreamURL(url, 'best')
    stream_data = cv2.VideoCapture(stream_url)
    return stream_data

def isTextPresent(image):
    """
    Credit to https://stackoverflow.com/questions/60906448/how-to-detect-text-using-opencv
    """
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Specify structure shape and kernel size. 
    # Kernel size increases or decreases the area 
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect 
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    
    # Appplying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE)
    
    # Creating a copy of image
    im2 = image.copy()

    print(contours)

def getBroadcastStart(stream_data):
    stream_fps = stream_data.get(cv2.CAP_PROP_FPS)
    stream_height = int(stream_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stream_width = int(stream_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    while True:
        ret, frame = stream_data.read()
        if ret:
            frame_number = int(stream_data.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            frame_interval = stream_fps / DATA_FPS
            if frame_number % frame_interval:
                # Get data from this frame
                y1 = int(stream_height - (stream_height / 5)) # The telemetry is on the bottom fifth of the screen, so calculate where that is based off the resolution
                cropped_frame = frame[0:stream_width, y1:stream_height]
                grayscale_image = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                print(pytesseract.image_to_string(grayscale_image))
                print(int(stream_data.get(cv2.CAP_PROP_POS_MSEC))/1000)
        else:
            break

def main():
    stream_data = getStreamData("https://www.youtube.com/watch?v=QJXxVtp3KqI")
    getBroadcastStart(stream_data)

if __name__ == '__main__':
    main()