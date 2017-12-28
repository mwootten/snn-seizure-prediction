# Based on specification at:
# http://www.mayo.edu/research/documents/mef-formatpdf/doc-10027429

import sys
import struct


def readString(bytes):
    return str(bytes, 'ascii').split('\x00')[0]


mefFile = sys.argv[1]
f = open(mefFile, 'rb')
header = f.read(1024)
f.close()

endianFlag = '>' if (header[163] == 0) else '<'
print('Endianness: {}'.format(endianFlag))

initial = struct.unpack('64s64s32sBB', header[0:162])

institution = readString(initial[0])
generalUse = readString(initial[1])
encAlg = readString(initial[2])
subjectEncryption = initial[3]
sessionEncryption = initial[4]

numberEntries = header[368:368+8]
channelName = header[376:376+32]
recordingStart = header[408:408+8]
recordingEnd = 0

print('Institution: {}'.format(institution))
print('General use text field: {}'.format(generalUse))
print('Encryption algorithm: {}'.format(encAlg))
print('Subject encryption: {}'.format(subjectEncryption == 1))
print('Session encryption: {}'.format(sessionEncryption == 1))

print('')
print('--- BEGIN SESSION PROPERTIES ---')
print('')

sessionData = struct.unpack('Q32sQQ ddddd 32s128s128si', header[368:756])

print('Number of entries:            {}'.format(sessionData[0]))
print('Channel name:                 {}'.format(readString(sessionData[1])))
print('Recording start:              {}'.format(sessionData[2]))
print('Recording end:                {}'.format(sessionData[3]))
print('Sampling frequency:           {}'.format(sessionData[4]))
print('Low frequency filter:         {}'.format(sessionData[5]))
print('High frequency filter:        {}'.format(sessionData[6]))
print('Notch frequency filter:       {}'.format(sessionData[7]))
print('Voltage conversion factor:    {}'.format(sessionData[8]))
print('Acquisition system:           {}'.format(readString(sessionData[9])))
print('Channel comments:             {}'.format(readString(sessionData[10])))
print('Study comments:               {}'.format(readString(sessionData[11])))
print('Physical channel number:      {}'.format(sessionData[12]))
