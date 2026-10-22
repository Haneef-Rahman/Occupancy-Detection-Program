"""Native, resolution-independent controls; inference runs in one worker.

Tk calls belong exclusively to the main thread. Only sensor pixels are images.
"""
import copy
import queue
import threading
import math


class DesktopBridge:
    def __init__(self):
        self.commands=queue.Queue();self.stop=threading.Event();self.lock=threading.Lock()
        self.state=dict(version=0,frame_version=0,ack=0,done=False,error='',message='Loading model…',row=None,data=None,controls=None,history=[])
        self.request_id=0

    def submit(self,command):
        self.request_id+=1;self.commands.put((self.request_id,command));return self.request_id

    def publish(self,ui=None,row=None,data=None,ack=None,**values):
        with self.lock:
            if ui is not None:
                d=ui.session.detector
                values['controls']=dict(conf=d.conf,low_conf=d.low_conf,mode=ui.session.mode,paused=ui.paused,
                    recording=bool(ui.recorder.continuous),saved=ui.recorder.total)
                values['history']=[dict(v) for v in ui.history.samples]
                values['message']=ui.message
            if row is not None:
                values['row']=copy.deepcopy(row);values['data']=data.copy();values['data'].flags.writeable=False
                self.state['frame_version']+=1
            if ack is not None:values['ack']=ack
            self.state.update(values);self.state['version']+=1

    def snapshot(self):
        with self.lock:return dict(self.state)

    def start(self,target):
        def work():
            try:target(self)
            except BaseException as error:self.publish(error=f'{type(error).__name__}: {error}')
            finally:self.publish(done=True)
        self.thread=threading.Thread(target=work,name='occupancy-inference',daemon=True);self.thread.start()


class NativeLauncher:
    def __init__(self,root,bridge,conf=.374,low_conf=.18,mode='auto',show_search=False,scale=5,live=True):
        import tkinter as tk
        from launcher_canvas import Dashboard
        self.root=root;self.bridge=bridge;self.live=live
        self.closed=False;self.closing=False;self.pending=0;self.version=-1;self.frame_version=-1
        self.state=bridge.snapshot();self.row=None;self.data=None;self.redraw=True
        self.palette=0;self.contrast=0;self.boxes=True;self.ids=True;self.raw=False;self.circles=show_search;self.history_visible=True
        self.paused=False;self.recording=False;self.mode=mode;self.saved=0;self.help_visible=False
        self.dragging=None;self.pointer=None;self.hover_button=None;self.pressed=None;self.chart_hover=None
        self.bubble_name=None;self.bubble_until=0;self.focused=None
        self.buttons={};self.sliders={}
        self.values={'conf':tk.DoubleVar(root,value=conf),'low_conf':tk.DoubleVar(root,value=low_conf)}
        self.status=tk.StringVar(root,value='Loading model…')
        self.status.trace_add('write',lambda *_:setattr(self,'redraw',True))
        root.title('FLUXNET — Occupancy workspace');root.configure(bg='#090909')
        root.geometry(f'{max(1120,min(1440,160*scale+440))}x840');root.minsize(1120,840)
        self.canvas=tk.Canvas(root,bg='#090909',highlightthickness=0,takefocus=True)
        self.canvas.pack(fill='both',expand=True);self.dashboard=Dashboard(self.canvas,self)
        self.canvas.bind('<Configure>',lambda _:setattr(self,'redraw',True))
        self.canvas.bind('<Motion>',self.motion);self.canvas.bind('<Leave>',self.leave)
        self.canvas.bind('<ButtonPress-1>',self.press);self.canvas.bind('<B1-Motion>',self.motion);self.canvas.bind('<ButtonRelease-1>',self.release)
        root.bind('<KeyPress>',self.key);root.protocol('WM_DELETE_WINDOW',self.close)
        self.canvas.focus_set();self.poll()

    def action(self,key):
        if self.closing:return
        if key=='q':self.close();return
        if key=='h':self.help_visible=not self.help_visible;self.redraw=True;return
        attr={'c':'circles','b':'boxes','i':'ids','g':'raw','o':'history_visible'}.get(key)
        if attr:setattr(self,attr,not getattr(self,attr));self.redraw=True
        elif key=='p':self.palette=(self.palette+1)%3;self.redraw=True
        elif key=='v':self.palette=0;self.contrast=(self.contrast+1)%3;self.redraw=True
        if self.state.get('done'):return
        if key in ('[',']',';',"'"):
            self.slider_key('conf' if key in ('[',']') else 'low_conf',.01 if key in (']',"'") else -.01);return
        elif key in ('1','2','3'):self.mode={'1':'auto','2':'precise','3':'faster'}[key]
        elif key=='t':self.mode=('auto','precise','faster')[(('auto','precise','faster').index(self.mode)+1)%3]
        elif key==' ':self.paused=not self.paused
        elif key=='n':self.paused=True
        if self.row is None and key in ('r','s'):self.status.set('Waiting for the first processed frame');return
        self.pending=self.bridge.submit(ord(key));self.redraw=True
        if not attr and key not in ('p','v'):self.status.set('Applying control after the current frame…')

    def key(self,event):
        if event.keysym=='Escape':self.close();return 'break'
        if event.keysym in ('Tab','ISO_Left_Tab'):
            keys=list(self.buttons)+list(self.sliders)
            i=keys.index(self.focused) if self.focused in keys else -1
            self.focused=keys[(i+(-1 if event.state & 1 else 1))%len(keys)];self.redraw=True;return 'break'
        if event.keysym in ('Left','Right') and self.focused in self.sliders:
            self.slider_key(self.focused,.01 if event.keysym=='Right' else -.01);return 'break'
        if event.keysym=='Return' and self.focused in self.buttons:self.action(self.focused);return 'break'
        if event.char and event.char in "qhbicgopv123t [];'nrxs":self.action(event.char);return 'break'

    @staticmethod
    def inside(p,b):return b[0]<=p[0]<=b[2] and b[1]<=p[1]<=b[3]

    def motion(self,event):
        self.pointer=(event.x,event.y)
        if self.dragging:self.slider_move(self.dragging,event.x);return
        button=next((k for k,b in self.buttons.items() if self.inside(self.pointer,b)),None)
        if button!=self.hover_button:self.hover_button=button;self.redraw=True
        slider=any(self.inside(self.pointer,s['bounds']) for s in self.sliders.values())
        self.canvas.configure(cursor='hand2' if button or slider else '')
        if not self.help_visible:self.dashboard.hover_chart(self.pointer)

    def leave(self,event):
        self.pointer=None;self.hover_button=None;self.redraw=True;self.dashboard.hover_chart(None)

    def press(self,event):
        self.canvas.focus_set();self.pointer=(event.x,event.y);self.focused=None
        if self.closing:return
        # The help overlay intercepts hits behind it; H still closes it.
        if self.help_visible:self.action('h');return
        if not self.state.get('done'):
            for name,s in self.sliders.items():
                if self.inside(self.pointer,s['bounds']):self.dragging=name;self.slider_move(name,event.x);return
        self.pressed=next((k for k,b in self.buttons.items() if self.inside(self.pointer,b)),None)

    def release(self,event):
        self.pointer=(event.x,event.y)
        if self.dragging:
            name=self.dragging;self.slider_move(name,event.x);self.dragging=None
            self.pending=self.bridge.submit(dict(slider=name,value=self.values[name].get()))
            self.status.set('Confidence applies after the current frame…');self.redraw=True;return
        if self.pressed and self.inside(self.pointer,self.buttons[self.pressed]):self.action(self.pressed)
        self.pressed=None

    def constrain(self,name,value):
        conf=self.values['conf'].get();low=self.values['low_conf'].get()
        return round(min(.99,max(low+.01,value)) if name=='conf' else min(conf-.01,max(.01,value)),3)

    def slider_move(self,name,x):
        x0,x1,_=self.sliders[name]['rail'];value=self.constrain(name,.01+.98*(x-x0)/(x1-x0))
        self.values[name].set(value);self.redraw=True

    def slider_key(self,name,delta):
        if self.closing or self.state.get('done'):return
        value=self.constrain(name,self.values[name].get()+delta);self.values[name].set(value)
        self.pending=self.bridge.submit(dict(slider=name,value=value));self.bubble_name=name;self.bubble_until=__import__('time').monotonic()+1
        self.status.set('Confidence applies after the current frame…');self.redraw=True

    def poll(self):
        if self.closed:return
        s=self.bridge.snapshot();self.state=s
        if s['version']!=self.version:
            self.version=s['version'];self.redraw=True
            self.status.set(s['error'] or ('Finished — inspect the last frame or quit' if s['done'] else s['message']))
            controls=s.get('controls')
            if controls and s['ack']>=self.pending:
                self.paused=controls['paused'];self.recording=controls['recording'];self.mode=controls['mode'];self.saved=controls['saved']
                for name in self.values:
                    if name!=self.dragging:self.values[name].set(controls[name])
            if s['frame_version']!=self.frame_version and s['row'] is not None:
                self.frame_version=s['frame_version'];self.row=s['row'];self.data=s['data']
        if self.bubble_name and __import__('time').monotonic()>=self.bubble_until:self.bubble_name=None;self.redraw=True
        if self.closing:
            self.status.set('Closing — finishing the current frame and saving files…')
            if s['done']:self.closed=True;self.root.destroy();return
        if self.redraw:
            self.dashboard.draw();self.redraw=False
            if self.focused:
                bounds=self.buttons.get(self.focused) or self.sliders.get(self.focused,{}).get('bounds')
                if bounds:self.canvas.create_rectangle(*bounds,outline='#eeeeee',dash=(2,3),width=1)
        self.root.after(16,self.poll)

    def close(self):
        if self.closing:return
        self.closing=True;self.bridge.stop.set();self.dragging=None


def run_desktop(target,args):
    import tkinter as tk
    root=tk.Tk();bridge=DesktopBridge()
    NativeLauncher(root,bridge,args.conf,args.recovery_conf,args.tracker,args.show_search,args.scale,not bool(args.replay))
    bridge.start(target)
    try:root.mainloop()
    finally:
        bridge.stop.set();bridge.thread.join(timeout=20)
    if bridge.snapshot()['error']:raise RuntimeError(bridge.snapshot()['error'])
