# coding=utf8
print('Starting diffsinger server...')
import json
import os
import sys
import traceback
import zmq

from inference.svs.ds_e2e import DiffSingerE2EInfer

notedict={
    0:"C",
    1:"C#",
    2:"D",
    3:"D#",
    4:"E",
    5:"F",
    6:"F#",
    7:"G",
    8:"G#",
    9:"A",
    10:"A#",
    11:"B"
}

def poll_socket(socket, timetick = 100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        pass
    # Escape while loop if there's a keyboard interrupt.

def readblocks(ustfile):
    #迭代器，从ust文件中逐个读取块，以字典形式返回
    current_block_data=dict()
    for line in ustfile.readlines():
        if(line.startswith("[")):#块的开始
            #先返回上一个
            if(len(current_block_data)>=1):
                yield current_block_data
            #然后开新块
            current_block_data={"name":line.strip("[#]\n")}
            pass
        else:
            (key,value)=line.strip("\n").split("=")
            current_block_data[key]=value
            pass
    if(len(current_block_data)>=1):#返回最后一个
        yield current_block_data

def acoustic(ustpath:str):
    #解析ust文件为diffsinger所需格式
    #参考：main.py
    tempo=120
    project=""
    voiceDir=""
    cacheDir=""
    
    ph_seq=[]
    note_seq=[]
    ph_dur=[]
    is_slur_seq=[]

    with open(ustpath) as ustfile:
        for block in readblocks(ustfile):
            if(block["name"]=="SETTING"):#音轨信息块
                tempo=float(block["Tempo"])
                project=block["Project"]
                voiceDir=block["VoiceDir"]
                cacheDir=block["CacheDir"]
            elif(block["name"].isdigit()):#音符
                lyric=block["Lyric"]
                notenum=int(block["NoteNum"])
                length=int(block["Length"])
                if(lyric!="R"):
                    ph_seq.append(lyric)
                    note_seq.append(notedict[notenum%12]+str(notenum//12-1))
                    ph_dur.append(str(length/(tempo*8)))
                    is_slur_seq.append("0")
    
    inp={
        "text":"",
        "ph_seq":" ".join(ph_seq),
        "note_seq":" ".join(note_seq),
        "ph_dur":" ".join(ph_dur),
        "note_dur_seq":" ".join(ph_dur),
        "is_slur_seq":" ".join(is_slur_seq),
        'input_type': 'phoneme'
    }
    #合成
    DiffSingerE2EInfer.example_run(inp, target=ustpath[:-4]+".wav")
    
#为了方便调试，把argv配置放外面
root_dir = os.path.dirname(__file__)
sys.argv = [
f'{root_dir}/inference/svs/ds_e2e.py',
'--config',
f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
'--exp_name',
'0228_opencpop_ds100_rel']

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind('tcp://*:38442')
    print('Started diffsinger server')

    for message in poll_socket(socket):
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        try:
            #if request[0] == 'timing':
            #    response['result'] = timing(request[1])
            if request[0] == 'acoustic':
                response['result'] = acoustic(request[1])
            else:
                raise NotImplementedError('unexpected command %s' % request[0])
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))

if(__name__=="__main__"):
    main()