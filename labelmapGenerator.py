import os
import sys
import re
from pathDefine import TFDir, datasetDir
from absl import flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('new', None, 'Create new file')
flags.DEFINE_string('add', None, 'Object class to be added')
flags.DEFINE_string('remove', None, 'Object class to be removed')
flags.DEFINE_boolean('listall', False, 'Lists all object classes')
FLAGS(sys.argv)



# if label map doesn't exist, create it
if not os.path.exists('label_map.pbtxt'):
  fileWrite = open('label_map.pbtxt', 'w')
  fileWrite.close()

entries = 0


# "add" flag code
if FLAGS.add:  # from command line, type: --add "object1, object2"
  results = [x for x in FLAGS.add.split(', ')] # split objects using commas as delimiters
  exists = []
  index = 0
  
  fileRead = open('label_map.pbtxt', 'r')
  for line in fileRead: # determine number of classes in label map
    if line == ("}\n"):
      entries += 1
  fileRead.close()
  

  # uses a check: this initializes each entry of "exists" to false.
  #  if one of the objects being added is already in the label map,
  #  exists will be set to true for that object, and the duplicate
  #  won't be added
  for index in range(len(results)):
    exists.append(False)
    print(results[index])
  
  for index in range(len(results)):
    fileRead = open('label_map.pbtxt', 'r')
    
    # iterate through lines
    for line in fileRead: 
      # detect duplicates
      if line == ("      object: " + results[index] + "\n"):
        exists[index] = True
        print('This object is already in the label map.')
        break 
    fileRead.close()
 
    # if this object isn't a duplicate, add it to the label map
    if exists[index] == False:
      entries += 1
      fileWrite = open('label_map.pbtxt', 'a')
      fileWrite.write("item {\n      id: %d\n      object: %s\n}\n" % (entries, results[index]))
      fileWrite.close()
  

if FLAGS.remove: # from command line, type: --remove "object1, object2"
  results = [x for x in FLAGS.remove.split(', ')]
  exists = []
  entries = 1 
  index = 0
  i = 0

  for index in range(len(results)):
    fileRead = open('label_map.pbtxt', 'r')
    lines = fileRead.readlines()
    fileRead.close()
    
    entries = 1
    fileWrite = open('label_map.pbtxt', 'w')
    for i in range(int(len(lines)/4)):
      # check every four lines (starting with line 3)
      #   this contains the object class name
      #   if there's a match, remove the neighboring four lines, i.e.:
      #   {
      #    id: 
      #    item:
      #   }
      if lines[4*i+2] != ("      object: " + results[index] + "\n"):
        fileWrite.write(lines[4*i])
        idstr = '      id: ' + str(entries) + '\n'
        fileWrite.write(idstr)
        fileWrite.write(lines[4*i+2])
        fileWrite.write(lines[4*i+3])
        entries += 1

    fileWrite.close()
    

if FLAGS.listall: # from command line, type: --listall
  fileRead = open('label_map.pbtxt', 'r')
  lines = fileRead.readlines()
  
  # check every four lines (starting with line 3)
  #   split based on colon and new line delimiters to isolate
  #   the object class name. Print this to the command line
  for i in range(int(len(lines)/4)):
    obj = re.split(': |\n', lines[4*i+2]) 
    print('id ' + str(i+1)+ ': ' + obj[1])
  fileRead.close()
