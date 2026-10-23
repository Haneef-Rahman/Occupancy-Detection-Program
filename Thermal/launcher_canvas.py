"""Reference-matched dashboard drawn with native Tk text and vector controls.

Only the 160x120 sensor image is rasterised. No screenshot APIs or web server.
"""
import math
import time
from functools import lru_cache

BG='#090909';CARD='#111111';BORDER='#2b2b2b';TEXT='#f2f2f2';MUTED='#919191'

class Dashboard:
    def __init__(self,canvas,ui):
        self.c=canvas;self.ui=ui;self.photo=None;self.image_key=None
        self.image_bounds=None;self.chart_bounds=None;self.chart_points=[]

    @lru_cache(maxsize=32)
    def font(self,size=14,bold=False):
        from tkinter.font import Font
        return Font(root=self.c,family='Arial',size=-size,weight='bold' if bold else 'normal')

    def text(self,label,x,y,size=14,color=TEXT,bold=False,**kw):
        return self.c.create_text(x,y,text=str(label),font=self.font(size,bold),fill=color,anchor='nw',**kw)

    def rounded(self,x,y,w,h,r=16,fill=CARD,outline=BORDER,tags=()):
        r=min(r,w/2,h/2)
        points=[]
        for cx,cy,start in [(x+w-r,y+r,-90),(x+w-r,y+h-r,0),(x+r,y+h-r,90),(x+r,y+r,180)]:
            for i in range(9):
                theta=math.radians(start+i*90/8)
                points.extend((cx+r*math.cos(theta),cy+r*math.sin(theta)))
        return self.c.create_polygon(points,fill=fill,outline=outline,width=1,tags=tags)

    def badge(self,label,x,y):
        w=self.font(11,True).measure(label)+20
        self.rounded(x,y,w,24,8,'#292929','');self.text(label,x+10,y+5,11,bold=True)

    def button(self,label,key,x,y,w,h=34,active=False,primary=False):
        u=self.ui;hover=u.pointer and x<=u.pointer[0]<=x+w and y<=u.pointer[1]<=y+h
        fill='#e5e5e5' if primary else '#292929' if active or hover else '#191919'
        outline='#ededed' if active else '#f2f2f2' if primary else '#444444' if hover else BORDER
        self.rounded(x,y,w,h,9,fill,outline)
        self.text(label,x+11,y+(h-16)//2,13,'#111111' if primary else TEXT,primary)
        shortcut='' if key.startswith('@') else 'SPACE' if key==' ' else key.upper();kw=max(20,self.font(9).measure(shortcut)+10)
        if shortcut:self.rounded(x+w-kw-7,y+7,kw,h-14,4,'#c9c9c9' if primary else '#303030','')
        if shortcut:self.text(shortcut,x+w-kw-2,y+11,9,'#222222' if primary else '#dddddd')
        u.buttons[key]=(x,y,x+w,y+h)

    def slider(self,name,label,y,right,rw,keys):
        u=self.ui;v=u.values[name].get()
        self.text(label,right+16,y,12,MUTED);self.text(f'{v:.3f}',right+rw-62,y,13,bold=True)
        x0=right+24;x1=right+rw-24;cy=y+49
        u.sliders[name]=dict(bounds=(x0-10,cy-16,x1+10,cy+16),rail=(x0,x1,cy))
        self.rounded(x0,cy-4,x1-x0,8,4,'#2b2b2b','')
        cx=x0+(v-.01)/.98*(x1-x0)
        self.rounded(x0,cy-4,max(4,cx-x0),8,4,'#e5e5e5','')
        held=u.dragging==name
        if held:self.c.create_oval(cx-15,cy-15,cx+15,cy+15,fill='#505050',outline='')
        r=8 if held else 10
        self.c.create_oval(cx-r,cy-r,cx+r,cy+r,fill=TEXT,outline=BG,width=2)
        if held or (u.bubble_name==name and time.monotonic()<u.bubble_until):
            bx=max(x0,min(x1-56,cx-28));by=cy-31
            self.rounded(bx,by,56,24,8,'#333333','#666666')
            self.text(f'{v:.3f}',bx+10,by+5,12,bold=True)
        self.text(keys,right+16,y+66,10,MUTED)

    def history(self,row,bounds):
        x,y,w,h=bounds;c=self.c;u=self.ui
        self.rounded(x,y,w,h)
        self.text('Occupancy history',x+16,y+12,15,bold=True)
        self.text(f'Observed {len(row["observed"])}   ·   Unresolved {row["unresolved"]}   ·   {row["processing_ms"]:.0f} ms',x+16,y+36,11,MUTED)
        lx=x+w-252;c.create_line(lx,y+20,lx+20,y+20,fill='#eeeeee',width=2)
        self.text('Estimate',lx+26,y+13,11,'#dddddd')
        c.create_line(lx+116,y+20,lx+134,y+20,fill='#888888',dash=(5,4),width=2)
        self.text('Observed',lx+142,y+13,11,'#bbbbbb')
        active=u.tab=='simulator' and row.get('heat_active',False) and row['mode']=='faster'
        if active:
            for i,(field,label) in enumerate([('detected','Detected'),('raw_heat','rFₕ'),('heat','Fₕ')]):
                self.text(f'{label} {row[field]:.1f}',x+16+i*150,y+53,11,('#bbbbbb','#777777','#ffffff')[i])
        left=x+47;right=x+w-18;top=y+(80 if active else 66);bottom=y+h-42
        self.chart_bounds=(left,top,right,bottom);self.chart_points=[]
        samples=(u.state.get('overview') or u.state.get('history',[])) if u.axis_mode=='compressed' else u.state.get('history',[])
        maximum=max([2]+[max(s['estimate'],s['observed'],*[s.get(k) or 0 for k in ('detected','raw_heat','heat')] if active else [0]) for s in samples]);step=max(1,math.ceil(maximum/3));maximum=step*3
        if samples:
            t0=0. if u.axis_mode=='compressed' else samples[0]['t'];span=samples[-1]['t']-t0
            def px(t):return left+(t-t0)/span*(right-left) if span else right
            def py(n):return bottom-n/maximum*(bottom-top)
            # Thin vector strips reproduce the reference's white-to-black area.
            # Each step stays at its actual integer count; no smoothed overshoot.
            intervals=[(px(a['t']),px(b['t']),py(a['estimate'])) for a,b in zip(samples,samples[1:])]
            for y0 in range(round(top),round(bottom),3):
                y1=min(bottom,y0+3);shade=round(17+44*(bottom-y0)/(bottom-top));color=f'#{shade:02x}{shade:02x}{shade:02x}'
                runs=[]
                for xa,xb,ys in intervals:
                    ya=max(y0,ys)
                    if ya<y1 and xb>xa:
                        if runs and runs[-1][1]==xa and runs[-1][2]==ya:runs[-1]=(runs[-1][0],xb,ya)
                        else:runs.append((xa,xb,ya))
                for xa,xb,ya in runs:c.create_rectangle(xa,ya,xb,y1,fill=color,outline='')
            self.chart_points=[(px(s['t']),s) for s in samples];self.chart_max=maximum
            for field,color,dash in [('estimate','#eeeeee',None),('observed','#888888',(5,4))]:
                points=[(px(samples[0]['t']),py(samples[0][field]))]
                for s in samples[1:]:points.extend([(px(s['t']),points[-1][1]),(px(s['t']),py(s[field]))])
                if len(points)>1:c.create_line(*[v for p in points for v in p],fill=color,width=1,dash=dash)
                else:
                    xx,yy=points[0];c.create_oval(xx-2,yy-2,xx+2,yy+2,fill=color,outline='')
            if active:
                for field,color,dash in [('detected','#bbbbbb',(2,4)),('raw_heat','#777777',(7,3)),('heat','#ffffff',None)]:
                    segment=[]
                    def flush():
                        if len(segment)>1:c.create_line(*[v for pt in segment for v in pt],fill=color,width=2 if field=='heat' else 1,dash=dash)
                    for sample in samples:
                        value=sample.get(field) if sample.get('heat_active') else None
                        if value is None:flush();segment=[]
                        else:segment.append((px(sample['t']),py(value)))
                    flush()
            for i in (range(4) if span else [3]):
                xx=left+(right-left)*i/3;tt=t0+span*i/3;label=f'{tt:.1f}s' if span<10 else f'{tt:.0f}s'
                self.text(label,xx-min(24,self.font(10).measure(label)/2),bottom+7,10,MUTED)
        else:self.text('Waiting for processed frames',left+12,top+18,12,MUTED)
        for i in range(4):
            yy=bottom-(bottom-top)*i/3
            c.create_line(left,yy,right,yy,fill='#303030',dash=(2,5))
            self.text(i*step,left-33,yy-6,10,MUTED)
        self.text(('Compressed overview · decimated' if u.axis_mode=='compressed' else 'Moving window · last 120 frames')+'  ·  '+('Fₕ diagnostic blend' if active else 'Estimate: Precise legacy / Faster observed'),x+16,y+h-17,10,'#777777')

    def hover_chart(self,pointer):
        c=self.c;c.delete('hover');self.ui.chart_hover=None
        if not pointer or not self.chart_bounds or not self.chart_points:return
        left,top,right,bottom=self.chart_bounds;x,y=pointer
        if not(left<=x<=right and top<=y<=bottom):return
        xx,s=min(self.chart_points,key=lambda p:abs(p[0]-x));self.ui.chart_hover=s
        yy=bottom-s['estimate']/self.chart_max*(bottom-top)
        c.create_line(xx,top,xx,bottom,fill='#777777',dash=(2,4),tags='hover')
        c.create_oval(xx-4,yy-4,xx+4,yy+4,fill='#bbbbbb',outline='#090909',tags='hover')
        tx=max(left,min(right-188,xx+12));ty=max(top+3,min(bottom-82,yy-10))
        self.rounded(tx,ty,180,82,10,'#1c1c1c','#444444',tags='hover')
        self.text(f'{s["t"]:.2f}s · frame {s["frame"]}',tx+10,ty+9,12,bold=True,tags='hover')
        self.text(f'Estimate {s["estimate"]}   Observed {s["observed"]}',tx+10,ty+32,11,'#dddddd',tags='hover')
        self.text(s['mode'].capitalize()+' K',tx+10,ty+56,11,MUTED,tags='hover')
        if self.ui.row and self.ui.row.get('heat_active') and s.get('heat_active'):
            self.rounded(tx,ty+82,180,58,8,'#1c1c1c','#444444',tags='hover')
            self.text(f'Detected {s["detected"]:.1f} · rFₕ {s["raw_heat"]:.1f}',tx+10,ty+90,11,tags='hover')
            self.text(f'Fₕ {s["heat"]:.1f}',tx+10,ty+112,11,bold=True,tags='hover')

    def thermal(self,data,row,bounds):
        from PIL import Image,ImageTk,ImageDraw
        from dataset_pipeline import colorize_view
        from crowd_detector import thermal_image
        import cv2
        x,y,w,h=bounds;c=self.c;u=self.ui
        iw=max(1,int(min(w,h*4/3)));ih=max(1,round(iw*.75));ix=x+(w-iw)/2;iy=y+(h-ih)/2
        self.image_bounds=(ix,iy,iw,ih)
        key=(u.frame_version,u.palette,u.contrast,iw,ih)
        if key!=self.image_key:
            if u.palette==0:im=colorize_view(data,u.contrast)
            else:
                gray=thermal_image(data)[:,:,0]
                im=cv2.applyColorMap(gray,cv2.COLORMAP_VIRIDIS) if u.palette==1 else cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((iw,ih),Image.Resampling.NEAREST).convert('RGBA')
            mask=Image.new('L',(iw,ih));ImageDraw.Draw(mask).rounded_rectangle((0,0,iw-1,ih-1),radius=10,fill=255)
            im.putalpha(mask);self.photo=ImageTk.PhotoImage(im,master=c);self.image_key=key
        c.create_image(ix,iy,image=self.photo,anchor='nw')
        fx=iw/160;fy=ih/120
        def box(b,color,label=None):
            bx,by,bw,bh=b
            c.create_rectangle(ix+bx*fx,iy+by*fy,ix+(bx+bw)*fx,iy+(by+bh)*fy,outline=color,width=1)
            if label:self.text(label,max(ix,min(ix+bx*fx,ix+iw-80)),max(iy,iy+by*fy-14),11,color)
        if u.raw:
            for d in row['base_detections']:box(d['bbox'],'#46b4f0')
        if u.boxes:
            for t in row['observed']:box(t['bbox'],'#87e66e' if t['source']=='base' else '#ffd232',f'#{t["id"]} {t["conf"]:.2f}' if u.ids else None)
        if u.circles and row['mode']=='faster':
            for r in row['regions']:
                cx=ix+r['cx']*fx;cy=iy+r['cy']*fy;rx=r['radius']*fx;ry=r['radius']*fy
                c.create_oval(cx-rx,cy-ry,cx+rx,cy+ry,outline='#64afdc')
                self.text(f'search #{r["id"]}',cx,cy,10,'#64afdc')

    def draw(self):
        from dataset_pipeline import VIEWS
        c=self.c;u=self.ui;W=max(1120,c.winfo_width());H=max(840,c.winfo_height())
        c.delete('all');c.create_rectangle(0,0,W,H,fill=BG,outline='');u.buttons={};u.sliders={};self.chart_bounds=None;self.chart_points=[]
        row=u.row or dict(observed=[],unresolved=0,processing_ms=0,mode=u.mode if u.mode!='auto' else 'precise',frame=0,estimate=0)
        self.button('Launcher','@launcher',24,16,112,active=u.tab=='launcher')
        self.button('Simulator','@simulator',146,16,112,active=u.tab=='simulator')
        from pathlib import Path
        self.text(str(Path(__file__).resolve().parent),24,55,11,MUTED)
        self.button('History','o',W-494,24,100,active=u.history_visible)
        self.button('Time: '+u.axis_mode,'@axis',W-382,24,140)
        if u.tab=='simulator':self.button('Heat settings','@heat',W-232,24,104)
        self.button('Quit','q',W-122,24,98);c.create_line(24,76,W-24,76,fill=BORDER)
        left=24;lw=W-408;right=W-360;rw=336
        if u.history_visible:self.history(row,(left,92,lw,192));thermal_top=298
        else:
            mw=(lw-24)/3
            for i,(label,value,hint) in enumerate([('Observed',len(row['observed']),'Current tracked detections'),('Unresolved',row['unresolved'],'Not added to crowd count'),('Processing',f'{row["processing_ms"]:.0f} ms','Detection + tracking')]):
                x=left+i*(mw+12);self.rounded(x,92,mw,82);self.text(label,x+16,104,12,MUTED);self.text(value,x+16,122,25,bold=True);self.text(hint,x+16,153,10,MUTED)
            thermal_top=188
        bottom=H-106;self.rounded(left,thermal_top,lw,bottom-thermal_top)
        self.text('Thermal view' if u.tab=='launcher' else '',left+18,thermal_top+16,15,bold=True)
        label=VIEWS[u.contrast][0] if u.palette==0 else 'fixed 15–45°C'
        self.text(f'160 × 120  /  '+['Inferno','Viridis','Grayscale'][u.palette]+' · '+label,left+18,thermal_top+(49 if u.tab=='simulator' else 39),11,MUTED)
        self.badge(row['mode'].capitalize()+' K' if u.row else 'Waiting',left+lw-116,thermal_top+15)
        analytics_x=left+16;analytics_y=thermal_top+66
        metrics=[('Detector',f'{row.get("base_ms",0):.1f} ms'),('Local recovery',f'{row.get("recovery_ms",0):.1f} ms'),('Tracking',f'{row.get("tracking_ms",0):.1f} ms'),('Mean score',f'{sum(t.get("conf",0) for t in row["observed"])/len(row["observed"]):.3f}' if row['observed'] else '—'),('Recovered now',str(sum(t.get('source')!='base' for t in row['observed'])))]
        for i,(label,value) in enumerate(metrics):
            yy=analytics_y+i*55;self.rounded(analytics_x,yy,142,48,9,'#181818','#282828')
            self.text(label,analytics_x+10,yy+6,10,MUTED);self.text(value,analytics_x+10,yy+22,15,bold=True)
        if u.data is not None:self.thermal(u.data,row,(left+174,thermal_top+62,lw-190,bottom-thermal_top-(96 if u.tab=='simulator' else 70)))
        else:self.text('Open a capture and press Play' if u.tab=='simulator' else 'Waiting for a thermal frame…',left+190,thermal_top+100,14,MUTED)
        py=H-92;self.rounded(left,py,lw,58)
        u.seek_bounds=None
        if u.tab=='simulator':
            # Timeline sits at the base of the thermal card, away from controls.
            x0=left+180;x1=left+lw-25;cy=bottom-16
            u.seek_bounds=(x0,cy-9,x1,cy+9)
            total=max(1,u.state.get('total',1));index=max(0,u.state.get('index',-1));fraction=index/max(1,total-1)
            if u.seek_drag:fraction=max(0,min(1,(u.seek_position-x0)/(x1-x0)))
            self.rounded(x0,cy-3,x1-x0,6,3,'#333333','');cx=x0+(x1-x0)*fraction
            c.create_oval(cx-6,cy-6,cx+6,cy+6,fill=TEXT,outline=BG)
            self.text(f'{index+1 if u.state.get("total") else 0} / {u.state.get("total",0)}',left+18,bottom-27,11,MUTED)

        self.button(('Play' if u.paused else 'Pause') if u.tab=='simulator' else 'Resume' if u.paused else 'Pause',' ',left+12,py+12,140,primary=True)
        self.button('Step','n',left+162,py+12,87);self.button('Reset','x',left+259,py+12,98);self.button('Help','h',left+367,py+12,90,active=u.help_visible)
        self.text(f'FRAME {row["frame"]:06d}',left+lw-144,py+14,12,MUTED)
        self.text(row['mode'].capitalize()+' K'+(f' · estimate {row["estimate"]}' if row['mode']=='precise' else ''),left+lw-144,py+32,10,MUTED)
        if u.tab=='simulator':
            self.button('Open capture','@open',left+18,thermal_top+14,126)
            self.button(f'{u.state.get("speed",1):g}× speed','@speed',left+154,thermal_top+14,100)
            self.button('Back','@back',left+264,thermal_top+14,70)
            self.button('Export','@export',left+344,thermal_top+14,78)
            self.button('Options','@options',left+432,thermal_top+14,82)
        self.rounded(right,92,rw,212);self.text('Detection settings',right+16,108,15,bold=True)
        self.slider('conf','Detection confidence',132,right,rw,'[  decrease     ]  increase')
        self.slider('low_conf','Local search confidence',216,right,rw,";  decrease     '  increase   ·   inside search circles only")
        self.rounded(right,318,rw,116);self.text('Tracking',right+16,333,15,bold=True);self.text('Choose a policy. Auto adapts to crowding.',right+16,357,11,MUTED)
        for i,(label,key,mode) in enumerate([('Auto','1','auto'),('Precise','2','precise'),('Faster','3','faster')]):self.button(label,key,right+16+i*102,382,96,active=u.mode==mode)
        self.rounded(right,448,rw,166);self.text('Dataset recording',right+16,464,15,bold=True);self.badge(str(u.saved)+' saved',right+rw-100,459)
        self.button('Stop recording' if u.recording else 'Start recording','r',right+16,496,rw-32,primary=True)
        self.button('Save single frame','s',right+16,540,rw-32);self.text('Thermal + YOLO labels  ·  original capture format',right+16,590,11,MUTED)
        self.rounded(right,628,rw,H-652);self.text('Display',right+16,644,15,bold=True)
        for i,(label,key,active) in enumerate([('Search circles','c',u.circles),('Boxes','b',u.boxes),('IDs / scores','i',u.ids),('Raw detections','g',u.raw),('Palette','p',False),('Contrast','v',u.palette==0)]):self.button(label,key,right+16+(i%2)*156,674+(i//2)*44,148,active=active)
        status=u.status.get()
        while status and self.font(12).measure(status)>W-56:status=status[:-2]
        self.text(status,28,H-21,12,'#bbbbbb')
        if u.help_visible:
            hx=left+28;hy=276;hw=lw-56
            self.rounded(hx,hy,hw,258,14,'#181818','#555555');self.text('Keyboard shortcuts',hx+20,hy+20,19,bold=True)
            for i,line in enumerate(["[ / ] Detection confidence     ; / ' Search confidence",'1 / 2 / 3 Auto / Precise / Faster     T Cycle tracker','R Record / stop    S Save one frame    X Reset tracker','Space Pause / resume    N Step    C Search circles','B Boxes    I IDs    G Raw detections    P Palette','V Contrast    O History    H Close help    Q Quit']):self.text(line,hx+20,hy+62+i*27,12,'#cccccc')
            self.text('Confidence changes take effect on the next processed frame.',hx+20,hy+232,11,MUTED)
        else:self.hover_chart(u.pointer)
