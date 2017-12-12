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
sampleDuration = 10 * seconds


def usecsToBytes(usecs):
    return 4 * int(round(usecs / eegLengthUsecs * channelFileLengthDwords))


f = open(channelFile, 'rb')

for (num, seizureStartUsecs) in enumerate(seizureStartsUsecs):
    sampleStartUsecs = seizureStartUsecs - sampleStartOffset

    sampleStartBytes = usecsToBytes(sampleStartUsecs)
    sampleDurationBytes = usecsToBytes(sampleDuration)

    print([sampleStartBytes, sampleDurationBytes])

    f.seek(sampleStartBytes)
    sample = f.read(sampleDurationBytes)
    outputName = channelFile.split('.')[0] + "-positive-" + str(num) + ".raw32"
    g = open(outputName, 'wb')
    g.write(sample)
    g.close()

f.close()
