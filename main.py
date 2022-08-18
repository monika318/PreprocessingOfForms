import cv2
import numpy as np
import pytesseract
import os

per = 70
pixelThreshold = 150

# roi=[[(218, 834), (510, 870), 'text', 'rollno.']]
roi=[[(108, 412), (268, 438), 'text', 'rollno'],
    [(300, 412), (450, 438), 'text', 'Rank'],
    [(482, 412), (642, 438), 'text', 'Score'],
    [(134, 478), (260, 504), 'text', 'SymbolNo.'], 
    [(318, 478), (462, 500), 'text', 'PassedYEAR2'], 
    [(550, 478), (652, 500), 'text', 'GPA2'],
    [(134, 503), (260, 528), 'text', 'Board2'], 
    [(490, 503), (652, 528), 'text', 'ExtraMaths'],
    [(150, 530), (652, 546), 'text', 'Institite/Colledge'],
    [(150, 560), (165, 575), 'Box', 'GovermentSEE'], 
    [(281, 560), (296, 575), 'Box', 'PrivateSEE'],
    [(134, 588), (260, 610), 'text', 'SymbolNo1'], 
    [(320, 588), (462, 610), 'text', 'Passedyear1'], 
    [(550, 588), (652, 610), 'text', 'GPA/PER1'],
    [(134, 614), (260, 640), 'text', 'Board1'], 
    [(318, 614), (652, 640), 'text', 'School'],
    [(165, 734), (652, 760), 'text', 'Name'], 
    [(165, 784), (340, 810), 'text', 'DOB'], 
    [(450, 784), (630, 810), 'text', 'DOBAD'],
    [(170, 817), (190, 834), 'Box', 'Male'], 
    [(226, 817), (246, 834), 'Box', 'Female'],
    [(132, 850), (275, 876), 'text', 'Contact'], 
    [(370, 850), (652, 876), 'text', 'Email'],
    [(192, 895), (365, 921), 'text', 'Municipality'], 
    [(425, 895), (465, 921), 'text', 'Wardno'], 
    [(485, 895), (652, 921), 'text', 'telno'], 
    [(90, 927), (250, 953), 'text', 'disctrict'], 
    [(300, 927), (460,953), 'text', 'Province'], 
    [(495, 927), (652, 953), 'text', 'Zone'],
    [(45, 258), (67, 282), 'Box', 'Civil'], 
    [(220, 258), (242, 282), 'Box', 'Electrical'], 
    [(45, 283), (67, 307), 'Box', 'Computer'], 
    [(220, 283), (242, 307), 'Box', 'Electronics'], 
    [(550, 177), (652, 310), 'Imahe', 'Photo']]


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

#Model = 'C:\\Users\\Monika\\Desktop\\DIGIT_Model_Development\\my_model.h5'

imgQ = cv2.imread('Query.jpg')
h, w, c = imgQ.shape

imgQ = cv2.resize(imgQ,(w//7,h//7),interpolation = cv2.INTER_AREA)

#cv2.imshow("query",imgQ)

orb = cv2.ORB_create(7000)
kp1, des1 = orb.detectAndCompute(imgQ, None)
impkp1 = cv2.drawKeypoints(imgQ,kp1,None)
cv2.imshow("key points",impkp1)

path = 'UserForms'
myPicList = os.listdir(path)
print(myPicList)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    #img = cv2.resize(img, (w//7, h//7))
    #cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
    matches = bf.match(des2, des1)
    matches = sorted(matches, key=lambda x: x.distance)
    # good=[]
    # for m,n in matches:
    #     if m.distance <0.7*n.distance:
    #         good.apppend(m)

    good = matches[:int(len(matches)*(per/100))]
    #<---------------condtion if the form is ours or not----------->
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:400], None, flags=2)
    imgMatch = cv2.resize(imgMatch, (w//6, h//7))
    cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ =cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(img, M, (w//7, h//7))
    imgScan = cv2.resize(imgScan, (w//7, h//7))
    cv2.imshow(y + "lol", imgScan)
    
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)
     
    #cv2.imshow('mask',imgMask)


    myData = []
    print(f'#################Extracting Data from Form {j}###########')
    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow(str(x), imgCrop)
    

        # if r[2] == 'text':
        #     print('{}:{}'.format(r[3], pytesseract.image_to_string(imgCrop)))
        #     myData.append(pytesseract.image_to_string(imgCrop))

        # if r[2] == 'Box':
        #     imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGRA2GRAY)
        #     imgThresh = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)[1]
        #     totalPixels = cv2.countNonZero(imgThresh)
        #     #print(totalPixels)
        #     if totalPixels > pixelThreshold : totalPixels = 1;
        #     else:totalPixels = 0
        #     print(f'{r[3]}:{totalPixels}')
        #     myData.append(totalPixels)
       # cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        with open('output.csv', 'a+') as f:
            for data in myData:
                f.write((str(data)+','))
            f.write('\n')
    #cv2.imshow(y,imgShow)

cv2.waitKey(0)

