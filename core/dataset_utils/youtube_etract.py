"""Downloads YouTube8M Dataset files for a specific partition from a mirror.

This download script will be served from http://data.yt8m.org/download.py. The
partitions are 1/{frame_level,video_level}/{train,validate,test}

To run locally, do:
  cat download.py | partition=2/video/train mirror=us python

Or to download just 1/1000th of the data:
  cat download.py | shard=1,1000 partition=2/video/train mirror=us python
"""

import hashlib
import itertools
import json
import os
from six.moves import urllib
import sys

def LetterRange(start, end):
  return list(map(chr, range(ord(start), ord(end) + 1)))

VOCAB = LetterRange('a', 'z') + LetterRange('A', 'Z') + LetterRange('0', '9')

file_ids = [''.join(i) for i in itertools.product(VOCAB, repeat=2)]

file_index = {f: i for (i, f) in enumerate(file_ids)}

def md5sum(filename):
  """Computes the MD5 Hash for the contents of `filename`."""
  md5 = hashlib.md5()
  with open(filename, 'rb') as fin:
    for chunk in iter(lambda: fin.read(128 * md5.block_size), b''):
      md5.update(chunk)
  return md5.hexdigest()

def download_file(source_url, destination_path):
  """Downloads `source_url` onto `destionation_path`."""
  def _progress(count, block_size, total_size):
    sys.stderr.write('\r>> Downloading %s %.1f%%' % (
        source_url, float(count * block_size) / float(total_size) * 100.0))
    sys.stderr.flush()
  urllib.request.urlretrieve(source_url, destination_path, _progress)
  statinfo = os.stat(destination_path)
  print('Succesfully downloaded', destination_path, statinfo.st_size, 'bytes.')
  return destination_path


if __name__ == '__main__':
  if 'partition' not in os.environ:
    print >> sys.stderr, (
        'Must provide environment variable "partition". e.g. '
        '0/video_level/train')
    exit(1)
  if 'mirror' not in os.environ:
    print >> sys.stderr, (
        'Must provide environment variable "mirror". e.g. "us"')
    exit(1)

  partition = os.environ['partition']
  mirror = os.environ['mirror']
  partition_parts = partition.split('/')

  assert mirror in {'us', 'eu', 'asia'}
  assert len(partition_parts) == 3
  assert partition_parts[1] in {'video_level', 'frame_level', 'video', 'frame'}
  assert partition_parts[2] in {'train', 'test', 'validate'}

  plan_url = 'http://data.yt8m.org/{}/download_plans/{}_{}.json'.format(
      partition_parts[0], partition_parts[1], partition_parts[2])

  num_shards = 1
  shard_id = 1
  if 'shard' in os.environ:
    if ',' not in os.environ['shard']:
      print ('Optional environment variable "shards" must be "X,Y" if set, '
             'where the integer X, Y are used for sharding. The files will be '
             'deterministically sharded Y-way and the X-th shard will be '
             'downloaded. It must be 1 <= X <= Y')
      exit(1)

    shard_id, num_shards = os.environ['shard'].split(',')
    shard_id = int(shard_id)
    num_shards = int(num_shards)
    assert shard_id >= 1
    assert shard_id <= num_shards

  plan_filename = '%s_download_plan.json' % partition.replace('/', '_')

  if os.path.exists(plan_filename):
    print ('Resuming Download ...')
  else:
    print ('Starting fresh download in this directory. Please make sure you '
           'have >2TB of free disk space!')
    download_file(plan_url, plan_filename)

  download_plan = json.loads(open(plan_filename).read())

  files = [f for f in download_plan['files'].keys()
           if int(hashlib.md5(f.encode('utf-8')).hexdigest(), 16) % num_shards == shard_id - 1]

  print ('Files remaining %i' % len(files))
  for f in files:
    fname, ext = f.split('.')
    out_f = '%s%04i.%s' % (str(fname[:-2]) , file_index[str(fname[-2:])], ext)

    if os.path.exists(out_f) and md5sum(out_f) == download_plan['files'][f]:
      print('Skipping already downloaded file %s' % out_f)
      continue
    elif os.path.exists(f) and md5sum(f) == download_plan['files'][f]:
      print('Skipping already downloaded file %s' % f)
      continue
    print('Downloading: %s' % out_f)

    download_url = 'http://%s.data.yt8m.org/%s/%s' % (mirror, partition, f)

    download_file(download_url, out_f)
    if md5sum(out_f) == download_plan['files'][f]:
      print('Successfully downloaded %s\n\n' % out_f)
      del download_plan['files'][f]
      open(plan_filename, 'w').write(json.dumps(download_plan))
    else:
      print ('Error downloading %s. MD5 does not match!\n\n' % f)

  print ('All done. No more files to download.')