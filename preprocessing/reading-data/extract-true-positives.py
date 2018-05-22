import sys
import os
import xml.etree.ElementTree as ET

annotationFile = sys.argv[1]
channelFile = sys.argv[2]
eegLengthUsecs = int(sys.argv[3])  # in microseconds. Should be unnecessary.

channelFileLengthDwords = int(os.path.getsize(channelFile) / 4)

root = ET.parse(annotationFile).getroot()
seizureStartsUsecs = [int(elem.attrib['startOffsetUsecs']) for elem in root]

usecs = 1
seconds = 1000000 * usecs
minutes = 60 * seconds
# Time before the seizure's start when the sample starts
sampleStartOffset = 10 * minutes
# Duration of the sample
sampleDuration = 5 * minutes


def usecsToBytes(usecs):
    return 4 * int(round(usecs / eegLengthUsecs * channelFileLengthDwords))


f = open(channelFile, 'rb')


def writeSlice(channelHandle, outputName, sampleStart, sampleDuration):
    sampleStartBytes = usecsToBytes(sampleStartUsecs)
    sampleDurationBytes = usecsToBytes(sampleDuration)
    channelHandle.seek(usecsToBytes(sampleStart))
    sample = channelHandle.read(sampleDurationBytes)
    with open(outputName, 'wb') as g:
        g.write(sample)


for (num, seizureStartUsecs) in enumerate(seizureStartsUsecs):
    sampleStartUsecs = seizureStartUsecs - sampleStartOffset
    outputName = channelFile.split('.')[0] + "-positive-" + str(num).zfill(3) + ".raw32"
    writeSlice(f, outputName, sampleStartUsecs, sampleDuration)

f.close()
