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
                values['overview']=list(ui.history.overview)
                values['message']=ui.message
                if hasattr(ui,'heat_config'):values['heat_config']=dict(ui.heat_config)
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
        self.axis_mode='compressed';self.tab='launcher';self.launcher_bridge=bridge;self.launcher_live=live;self.sim_bridge=None;self.seek_drag=False
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
        root.title('Occupancy Detection — Thermal workspace');root.configure(bg='#090909')
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
        if key.startswith('@'):
            self.workspace_action(key);return
        if self.tab=='simulator' and key in (' ','n','x'):
            self.pending=self.bridge.submit(ord(key));return
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

    def workspace_action(self,key):
        if key in ('@launcher','@simulator'):
            target=key[1:]
            if target==self.tab:return
            if self.tab=='launcher' and not self.state.get('done') and not self.paused:self.launcher_bridge.submit(ord(' '))
            if self.tab=='simulator':self.sim_bridge.sim.submit({'action':'pause'})
            if target=='simulator' and self.sim_bridge is None:
                self.sim_bridge=SimulatorBridge()
            self.tab=target;self.bridge=self.launcher_bridge if target=='launcher' else self.sim_bridge
            self.live=self.launcher_live if target=='launcher' else False
            self.version=-1;self.frame_version=-1;self.pending=0;self.row=None;self.data=None;self.state=self.bridge.snapshot()
            self.dashboard.image_key=None;self.redraw=True;self.dragging=None
            self.paused=not self.state.get('playing',False) if target=='simulator' else (self.state.get('controls') or {}).get('paused',False)
            self.recording=False;self.saved=0
        elif key=='@axis':
            self.axis_mode='moving' if self.axis_mode=='compressed' else 'compressed';self.redraw=True
        elif key=='@heat':self.heat_settings()
        elif key=='@open':
            from tkinter import filedialog,simpledialog
            folder=filedialog.askdirectory(parent=self.root,title='Choose a logs/capture folder')
            if folder:
                try:
                    from capture_replay import CaptureSequence
                    CaptureSequence(folder);fps=None
                except Exception as error:
                    if 'timestamp' not in str(error).lower():self.status.set(str(error));return
                    fps=simpledialog.askfloat('Recorded frame rate','Timestamps missing. Enter the known recording FPS:',parent=self.root,minvalue=.001)
                    if fps is None:return
                self.bridge.sim.submit(dict(action='load',folder=folder,conf=self.values['conf'].get(),low_conf=self.values['low_conf'].get(),mode=self.mode,fps=fps))
        elif key=='@options':
            import tkinter as tk
            state=self.bridge.sim.snapshot();folder=state.get('capture',{}).get('folder')
            if not folder:self.status.set('Open a capture before changing replay options');return
            top=tk.Toplevel(self.root);top.title('Simulator options');top.configure(bg='#111111');top.transient(self.root)
            settings=state.get('settings',{});config=settings.get('config',{})
            variables={}
            fields=[('window','Recovery window (seconds)',config.get('recovery_seconds',2)),('radius','Search radius cap (pixels)',config.get('max_radius',14)),('max_speed','Speed gate (pixels/second)',config.get('max_speed',30)),('fps','FPS override (blank uses timestamps)','')]
            for i,(key,label,value) in enumerate(fields):
                tk.Label(top,text=label,bg='#111111',fg='#eeeeee').grid(row=i,column=0,padx=16,pady=10,sticky='w')
                variables[key]=tk.StringVar(top,value=str(value));tk.Entry(top,textvariable=variables[key],bg='#222222',fg='#eeeeee',insertbackground='white',width=12).grid(row=i,column=1,padx=16,pady=10)
            error=tk.StringVar(top)
            tk.Label(top,textvariable=error,bg='#111111',fg='#eeeeee').grid(row=4,column=0,columnspan=2)
            def apply():
                try:
                    values={k:float(v.get()) for k,v in variables.items() if v.get().strip()}
                    import math
                    if any(not math.isfinite(v) or v<=0 for v in values.values()):raise ValueError('Values must be finite and positive')
                    self.bridge.sim.submit(dict(action='load',folder=folder,conf=self.values['conf'].get(),low_conf=self.values['low_conf'].get(),mode=self.mode,**values))
                    top.destroy()
                except Exception as e:error.set(str(e))
            tk.Button(top,text='Apply and restart replay',command=apply,bg='#eeeeee',fg='#111111').grid(row=5,column=0,columnspan=2,padx=16,pady=16)
        elif key=='@speed':
            speeds=(.25,.5,1.,2.,4.,16.);speed=self.state.get('speed',1.)
            self.bridge.sim.submit(dict(action='speed',value=speeds[(speeds.index(speed)+1)%len(speeds)]))
        elif key=='@back':self.bridge.sim.submit(dict(action='seek',index=max(0,self.state.get('index',0)-1)))
        elif key=='@export':self.bridge.sim.submit(dict(action='export'))

    def heat_settings(self):
        import tkinter as tk
        from dataclasses import asdict
        from heat_function import HeatConfig
        target_bridge=self.bridge
        current=self.state.get('heat_config') or asdict(HeatConfig())
        top=tk.Toplevel(self.root);top.title('Heat diagnostic settings');top.configure(bg='#111111');top.transient(self.root)
        variables={}
        labels={'low_c':'Lower temperature °C (test band)','high_c':'Upper temperature °C (test band)','near_weight':'Near / bottom pixel weight','far_weight':'Far / top pixel weight','power':'Vertical gradient exponent','area':'Weighted pixels per occupancy unit','a':'a: raw heat weight','b':'b: detection baseline weight','warmup_s':'Continuous Faster K seconds','switch_window_s':'Recent switch window seconds','max_precise_entries':'Maximum Precise K entries'}
        for i,(key,label) in enumerate(labels.items()):
            tk.Label(top,text=label,bg='#111111',fg='#eeeeee').grid(row=i,column=0,padx=14,pady=3,sticky='w')
            variables[key]=tk.DoubleVar(top,value=current[key])
            ranges={'low_c':(0,45,.5),'high_c':(10,60,.5),'near_weight':(.1,10,.1),'far_weight':(.1,20,.1),'power':(.1,5,.1),'area':(1,3000,1),'a':(0,.49,.01),'b':(.51,1,.01),'warmup_s':(0,60,1),'switch_window_s':(1,120,1),'max_precise_entries':(0,10,1)}
            lo,hi,step=ranges[key]
            tk.Scale(top,variable=variables[key],from_=lo,to=hi,resolution=step,orient='horizontal',length=240,bg='#111111',fg='#eeeeee',troughcolor='#333333',highlightthickness=0).grid(row=i,column=1,padx=14)
        syncing=[False]
        def blend(source,target):
            if syncing[0]:return
            syncing[0]=True;variables[target].set(round(1-variables[source].get(),2));syncing[0]=False
        variables['a'].trace_add('write',lambda *_:blend('a','b'));variables['b'].trace_add('write',lambda *_:blend('b','a'))
        message=tk.StringVar(top,value='Test band only. Fₕ is floored; Detected = Fₕ + raw boxes without a visible tracking ID. pD holds the peak and follows whole rFₕ changes, never below current Detected.')
        tk.Label(top,textvariable=message,bg='#111111',fg='#bbbbbb',wraplength=510).grid(row=11,column=0,columnspan=2,pady=8)
        def apply():
            try:
                values={k:v.get() for k,v in variables.items()};values['max_precise_entries']=int(values['max_precise_entries'])
                config=HeatConfig(**values)
                target_bridge.submit(dict(action='heat_config',values=asdict(config)));top.destroy()
            except Exception as e:message.set(str(e))
        tk.Button(top,text='Apply for subsequent frames',command=apply,bg='#eeeeee',fg='#111111').grid(row=12,column=0,columnspan=2,pady=10)

    def key(self,event):
        if self.tab=='simulator' and self.focused not in self.sliders and event.keysym in ('Left','Right','Home','End'):
            if event.keysym=='Right':self.action('n')
            elif event.keysym=='Left':self.workspace_action('@back')
            else:self.bridge.sim.submit(dict(action='seek',index=0 if event.keysym=='Home' else max(0,self.state.get('total',1)-1)))
            return 'break'
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
        if self.seek_drag:self.seek_position=event.x;self.redraw=True;return
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
        if self.tab=='simulator' and getattr(self,'seek_bounds',None) and self.inside(self.pointer,self.seek_bounds):
            self.seek_drag=True;self.seek_position=event.x;self.redraw=True;return
        # The help overlay intercepts hits behind it; H still closes it.
        if self.help_visible:self.action('h');return
        if not self.state.get('done'):
            for name,s in self.sliders.items():
                if self.inside(self.pointer,s['bounds']):self.dragging=name;self.slider_move(name,event.x);return
        self.pressed=next((k for k,b in self.buttons.items() if self.inside(self.pointer,b)),None)

    def release(self,event):
        self.pointer=(event.x,event.y)
        if self.seek_drag:
            self.seek_drag=False;x0,y0,x1,y1=self.seek_bounds
            index=round(max(0,min(1,(event.x-x0)/(x1-x0)))*max(0,self.state.get('total',1)-1))
            self.bridge.sim.submit(dict(action='seek',index=index));self.redraw=True;return
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
            if s['row'] is None:self.row=None;self.data=None
            if s['frame_version']!=self.frame_version and s['row'] is not None:
                self.frame_version=s['frame_version'];self.row=s['row'];self.data=s['data']
        if self.bubble_name and __import__('time').monotonic()>=self.bubble_until:self.bubble_name=None;self.redraw=True
        if self.closing:
            self.status.set('Closing — finishing the current frame and saving files…')
            if self.launcher_bridge.snapshot()['done'] and (not self.sim_bridge or self.sim_bridge.sim.closed):self.closed=True;self.root.destroy();return
        if self.redraw:
            self.dashboard.draw();self.redraw=False
            if self.focused:
                bounds=self.buttons.get(self.focused) or self.sliders.get(self.focused,{}).get('bounds')
                if bounds:self.canvas.create_rectangle(*bounds,outline='#eeeeee',dash=(2,3),width=1)
        self.root.after(16,self.poll)

    def close(self):
        if self.closing:return
        self.closing=True;self.launcher_bridge.stop.set();self.dragging=None
        if self.sim_bridge:self.sim_bridge.stop.set();self.sim_bridge.sim.submit({'action':'shutdown'})


class SimulatorBridge:
    """Adapt the existing simulator worker to the shared native dashboard."""
    def __init__(self,sim=None):
        from pathlib import Path
        from model_registry import resolve_weights
        from tracking_simulator import Simulator
        root=Path(__file__).resolve().parent
        self.sim=sim or Simulator(root/'logs',root/'logs/tracking_simulations',resolve_weights(None),'cpu')
        self.sim.reset_global_ids=False
        self.stop=threading.Event();self.request_id=0
    def submit(self,command):
        self.request_id+=1
        if isinstance(command,dict) and command.get('action')=='heat_config':cmd=command
        elif isinstance(command,int) and chr(command) in ' nx':
            key=chr(command);action=('pause' if self.sim.snapshot()['playing'] else 'play') if key==' ' else 'step' if key=='n' else 'restart'
            cmd=dict(action=action)
        else:cmd=dict(action='control',command=command)
        self.sim.submit(dict(**cmd,native_request=self.request_id));return self.request_id
    def snapshot(self):
        s=self.sim.snapshot();controls=s.get('controls')
        if controls:controls=dict(controls,paused=not s['playing'])
        return dict(s,version=s.get('native_version',0),frame_version=s.get('native_frame_version',0),ack=s.get('native_ack',0),done=self.sim.closed,message=s['status'],row=s.get('row'),data=s.get('data'),controls=controls,history=s.get('history',[]),overview=s.get('overview',[]))

def run_desktop(target,args):
    import tkinter as tk
    root=tk.Tk();bridge=DesktopBridge()
    NativeLauncher(root,bridge,args.conf,args.recovery_conf,args.tracker,args.show_search,args.scale,not bool(args.replay))
    bridge.start(target)
    try:root.mainloop()
    finally:
        bridge.stop.set();bridge.thread.join(timeout=20)
    if bridge.snapshot()['error']:raise RuntimeError(bridge.snapshot()['error'])


def run_simulator_desktop(sim):
    import tkinter as tk
    root=tk.Tk();idle=DesktopBridge();idle.publish(done=True,message='Launcher idle — start integrated_launcher_v2.py for live capture')
    ui=NativeLauncher(root,idle,live=False)
    ui.sim_bridge=SimulatorBridge(sim);ui.workspace_action('@simulator')
    try:root.mainloop()
    finally:sim.close()
