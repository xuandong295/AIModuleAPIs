#import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
import numpy as np
import urllib
import json
import cv2
import os
from PIL import Image
import numpy as np
from requests import Response
from rest_framework.generics import ListAPIView
# define the path to the face detector
from rest_framework import views
from rest_framework.parsers import FileUploadParser, BaseParser, MultiPartParser

FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
from rest_framework.decorators import api_view
from rest_framework.decorators import parser_classes
from rest_framework.parsers import JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
@csrf_exempt
def detect(request):

    if request.method == "POST":
        # initialize the data dictionary to be returned by the request
        data = {"success": False}


        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])
        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("url", None)
            print(url)
            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)
            # load the image and convert
            image = _grab_image(url=url)
        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(image_gray, scaleFactor=1.1)
        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        # update the data dictionary with the faces detected
        for (x, y, w, h) in rects:
             cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)
        # cv2.imshow("Result", image)
        # cv2.waitKey(0)
        print(image.shape)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(imageRGB)
        img.save('my.png')
        img.show()
        data.update({"num_faces": len(rects), "faces": rects, "success": True})
        try:
            with open('my.png', "rb") as f:
                return HttpResponse(f.read(), content_type="image/jpeg")
        except IOError:
            red = Image.new('RGBA', (1, 1), (255, 0, 0, 0))
            response = HttpResponse(content_type="image/jpeg")
            red.save(response, "JPEG")
            return response
        # return a JSON response
        #return HttpResponse(img, content_type="image/png")
        return JsonResponse(data)


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image_imdecode = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image_imdecode


class FileUploadView(views.APIView):
    parser_classes = (MultiPartParser,)

    def post(self, request, filename):
        file_obj = request.FILES['file']
        destination = open(filename, 'wb+')
        for chunk in file_obj.chunks():
            destination.write(chunk)
        destination.close()  # File should be closed only after all chuns are added
        # do some stuff with uploaded file
        return Response(status=204)