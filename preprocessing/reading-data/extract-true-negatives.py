import sys
import os
import xml.etree.ElementTree as ET
import random

random.seed(5)

annotationFile = sys.argv[1]
channelFile = sys.argv[2]
eegLengthUsecs = int(sys.argv[3])  # in microseconds. Should be unnecessary.

channelFileLengthDwords = int(os.path.getsize(channelFile) / 4)

root = ET.parse(annotationFile).getroot()
seizureStartsUsecs = [int(elem.attrib['startOffsetUsecs']) for elem in root]
seizureEndsUsecs = [int(elem.attrib['endOffsetUsecs']) for elem in root]

def alternates(starts, ends):
    '''
    Verifies that the EEG starts and ends do alternate ---
    there are no overlapping seizure events. This should be true, but this
    check should save some debugging if it's ever false.
    '''
    taggedStarts = [(t, 'start') for t in starts]
    taggedEnds = [(t, 'end') for t in ends]
    allTagged = sorted(taggedStarts + taggedEnds)
    correctState = 'start'
    for (t, state) in allTagged:
        if state != correctState:
            return False
        if correctState == 'start':
            correctState = 'end'
        else:
            correctState = 'start'
    return (correctState == 'start')


if not alternates(seizureStartsUsecs, seizureEndsUsecs):
    print('Does not alternate between start and end - errors probable')
    sys.exit(1)

usecs = 1
seconds = 1000000 * usecs
minutes = 60 * seconds

excludeBefore = 30 * minutes
excludeAfter = 30 * minutes
sampleDuration = 5 * minutes

def usecsToBytes(usecs):
    return 4 * int(round(usecs / eegLengthUsecs * channelFileLengthDwords))

def writeSlice(channelHandle, outputName, sampleStart, sampleDuration):
    sampleStartBytes = usecsToBytes(sampleStart)
    sampleDurationBytes = usecsToBytes(sampleDuration)
    channelHandle.seek(sampleStartBytes)
    sample = channelHandle.read(sampleDurationBytes)
    with open(outputName, 'wb') as g:
        g.write(sample)

# The general gist of the algorithm is this: we want to exclude all times from
# excludeBefore minutes to excludeAfter minutes. That time makes up some
# intervals on the EEG. We want to select 10 second segments at random from the
# remaining slices. To do this, we make a series of intervals stretching from
# [time of last end, time of next start - sample duration]. Add up the lengths
# of all these segments, and pick a random number from [0, that sum]. Translate
# that index into the index in the file overall. Take that as the start time,
# and pick out the duration later. Copy the slices over, and we're done.

# Maybe exclude any future samples that overlap. I'm not sure how big of a deal
# this will end up being.


# Step 1: pick out the times where a slice *cannot* begin. Note that some of
# these times are outside of the exclusion zone; however, by the end of the
# sample it will have strayed into that zone. While with current parameter
# values this isn't a big deal, a significant increase of sampleDuration will
# make this very important.

unacceptableStartSlices = []

# This isn't used for a few more steps, but it is easiest to compute this here.
unacceptableSliceLengths = []

for (seizureStart, seizureEnd) in zip(seizureStartsUsecs, seizureEndsUsecs):
    # To see why these intervals are what they are, take the following:
    # t - some time being considered for the start of a sample
    # d - the duration of the sample
    # s - the start of the exclusion zone (start - excludeBefore)
    # e - the end of the exclusion zone (end + excludeAfter)
    #
    # We want the following two things to be true:
    # t     ∈ [s, e]    (the start is within bounds)
    # t + d ∈ [s, e]    (the end is within bounds)
    # The second equation then becomes:
    # t ∈ [s - d, e - d]
    # For t to be in both, it has to be in the intersection of those
    # two intervals. Therefore:
    # t ∈ [s, e] ∩ [s - d, e - d]
    # t ∈ [s, e - d]
    sliceStart = seizureStart - excludeBefore
    sliceEnd = seizureEnd + excludeAfter - sampleDuration

    # We add these in a somewhat peculiar fashion. Instead of adding a tuple,
    # we add the start, then the end, making no distinction other than order.
    # This turns out to be useful for the next step.

    unacceptableStartSlices.append(sliceStart)
    unacceptableStartSlices.append(sliceEnd)

    unacceptableSliceLengths.append(sliceEnd - sliceStart)

# Step 2: Convert from a sequence of unacceptable slices into a list of acceptable ones.
# This algorithm works by adding the endpoints of the sequence, and then reading off
# pairs. For example, take the example of an EEG 8 units long. The interval [2, 3] is
# blocked off, as well as [5, 6]. Therefore, unacceptableStartSlices is:
# [2, 3, 5, 6]
# When we add the endpoints, we get:
# [0, 2, 3, 5, 6, 8]
# Reading off in pairs, we get valid zones as [0, 2], [3, 5], and [6, 8]. Verifying
# manually, this is precisely what we would expect.

acceptableStartSlices = unacceptableStartSlices[:]
acceptableStartSlices.insert(0, 0)
acceptableStartSlices.append(eegLengthUsecs)

acceptableStartSlicePairs = []

for i in range(len(acceptableStartSlices) // 2):
    slice = (acceptableStartSlices[2 * i], acceptableStartSlices[2 * i + 1])
    acceptableStartSlicePairs.append(slice)

# Step 3: Find the total length of the acceptable regions

acceptableSliceLengths = [end - start for (start, end) in acceptableStartSlicePairs]
totalAcceptableLength = sum(acceptableSliceLengths)

def getAddend(index):
    '''
    Get the additional offset that must be added to convert an index into just
    acceptable regions into an index into the entire EEG
    '''
    x = index
    a = acceptableSliceLengths
    u = unacceptableSliceLengths
    sa = a[0]
    su = 0
    i = 0
    # TODO: Should this be greater than or equal to?
    while x > sa:
        su += u[i]
        sa += a[i+1]
        i += 1
    return su

# The next three steps must be performed for each sample.

f = open(channelFile, 'rb')

negativesCount = len(seizureStartsUsecs)
for num in range(negativesCount):
    # Step 4: Pick an index within the acceptable regions uniformly at random
    indexWithinAcceptableRegions = random.uniform(0, totalAcceptableLength)

    # Step 5: Map this value onto the actual start value.
    # TODO: Explain how this works.
    actualSliceStart = indexWithinAcceptableRegions + getAddend(indexWithinAcceptableRegions)

    # Step 6: Write out the sample.
    outputName = outputName = channelFile.split('.')[0] + "-negative-" + str(num) + ".raw32"
    writeSlice(f, outputName, actualSliceStart, sampleDuration)
f.close()
