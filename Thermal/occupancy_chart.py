"""Bounded, timestamped occupancy history and monochrome area-chart rendering."""
from collections import deque
import math
import numpy as np
from PIL import Image, ImageDraw


class OccupancyHistory:
    def __init__(self,limit=120):
        self.samples=deque(maxlen=limit);self.start=None;self.last_key=None;self.overview=[]

    def clear(self):
        self.samples.clear();self.start=None;self.last_key=None;self.overview=[]

    def append(self,row):
        key=(row['frame'],row['time'],row.get('origin'))
        if key==self.last_key:return
        timestamp=float(row['time'])
        if not math.isfinite(timestamp):raise ValueError('invalid history timestamp')
        if self.samples and timestamp<=self.samples[-1]['timestamp']:self.clear()
        if self.start is None:self.start=timestamp
        self.samples.append(dict(frame=row['frame'],timestamp=timestamp,t=timestamp-self.start,
            estimate=int(row['estimate']),observed=len(row['observed']),mode=row['mode']))
        for name in ('heat_active','raw_heat','heat','detected'):
            self.samples[-1][name]=row.get(name)
        self.overview.append(dict(self.samples[-1]))
        if len(self.overview)>240:self.overview=self.overview[:-1:2]+[self.overview[-1]]
        self.last_key=key


def draw_history(surface,ui,row,bounds):
    from launcher_theme import font
    x,y,w,h=bounds;d=ImageDraw.Draw(surface)
    def text(s,xx,yy,size=11,color='#999999',bold=False):
        d.text((xx,yy),str(s),font=font(size,bold),fill=color)
    d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill='#111111',outline='#2b2b2b')
    text('Occupancy history',x+16,y+12,15,'#f2f2f2',True)
    text(f'Observed {len(row["observed"])}   ·   Unresolved {row["unresolved"]}   ·   {row["processing_ms"]:.0f} ms',x+16,y+36,11)
    # Form and luminance both distinguish the two traces.
    lx=x+w-252
    d.line((lx,y+20,lx+20,y+20),fill='#eeeeee',width=2);text('Estimate',lx+26,y+13,11,'#dddddd')
    for a in (lx+116,lx+125):d.line((a,y+20,a+5,y+20),fill='#888888',width=2)
    text('Observed',lx+142,y+13,11,'#bbbbbb')
    left=x+47;right=x+w-18;top=y+66;bottom=y+h-42
    ui.chart_bounds=(left,top,right,bottom);ui.chart_hover=None
    samples=list(ui.history.samples)
    maximum=max([2]+[max(s['estimate'],s['observed']) for s in samples])
    step=max(1,math.ceil(maximum/3));maximum=step*3
    for i in range(4):
        yy=round(bottom-(bottom-top)*i/3)
        for xx in range(left,right,7):d.line((xx,yy,min(xx+2,right),yy),fill='#303030')
        text(i*step,left-33,yy-6,10)
    if not samples:
        text('Waiting for processed frames',left+12,top+18,12)
        return
    t0=samples[0]['t'];t1=samples[-1]['t'];span=t1-t0
    def px(t):return left+(t-t0)/span*(right-left) if span>0 else right
    def py(n):return bottom-n/maximum*(bottom-top)
    for i in range(4):
        xx=round(left+i*(right-left)/3);tt=t0+span*i/3
        # Real elapsed seconds, not assumed frame rate or calendar placeholders.
        label=f'{tt:.1f}s' if span<10 else f'{tt:.0f}s'
        text(label,xx-min(24,round(d.textlength(label,font=font(10))/2)),bottom+7,10)
    def step_points(field):
        points=[(px(samples[0]['t']),py(samples[0][field]))]
        for s in samples[1:]:
            xx=px(s['t']);points.extend([(xx,points[-1][1]),(xx,py(s[field]))])
        return points
    estimate=step_points('estimate');observed=step_points('observed')
    if len(estimate)>1:
        # White-to-transparent area on a black card, like the supplied reference.
        mask=Image.new('L',surface.size,0);md=ImageDraw.Draw(mask)
        md.polygon([(estimate[0][0],bottom),*estimate,(estimate[-1][0],bottom)],fill=255)
        gradient=np.zeros((surface.height,surface.width,4),np.uint8)
        gradient[:,:,:3]=235
        for yy in range(top,bottom+1):gradient[yy,left:right+1,3]=round(65*(bottom-yy)/max(1,bottom-top)+5)
        alpha=Image.fromarray(gradient,'RGBA');alpha.putalpha(Image.fromarray(np.minimum(gradient[:,:,3],np.asarray(mask))))
        surface.paste(alpha,(0,0),alpha)
        d=ImageDraw.Draw(surface)
        d.line(estimate,fill='#eeeeee',width=2)
        # Dashed observed trace stays visible even when it coincides with estimate.
        for a,b in zip(observed,observed[1:]):
            length=math.hypot(b[0]-a[0],b[1]-a[1])
            for k in range(0,math.ceil(length),8):
                t=k/max(length,1);u=min(k+4,length)/max(length,1)
                d.line((a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u),fill='#888888',width=2)
    else:
        ex,ey=estimate[0];d.ellipse((ex-3,ey-3,ex+3,ey+3),fill='#eeeeee')
    text('Last 120 frames  ·  Estimate: Precise legacy / Faster observed',x+16,y+h-18,10,'#777777')
    if ui.pointer and left<=ui.pointer[0]<=right and top<=ui.pointer[1]<=bottom:
        s=min(samples,key=lambda s:abs(px(s['t'])-ui.pointer[0]));ui.chart_hover=dict(s)
        xx=round(px(s['t']))
        for yy in range(top,bottom,6):d.line((xx,yy,xx,min(yy+2,bottom)),fill='#666666')
        for field,color in [('estimate','#eeeeee'),('observed','#888888')]:
            yy=round(py(s[field]));d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color,outline='#090909',width=2)
        tx=max(left,min(right-184,xx+12));ty=top+4
        d.rounded_rectangle((tx,ty,tx+180,ty+83),radius=11,fill='#1c1c1c',outline='#414141')
        text(f'{s["t"]:.2f}s · frame {s["frame"]}',tx+12,ty+9,12,'#f2f2f2',True)
        text(f'Estimate {s["estimate"]}    Observed {s["observed"]}',tx+12,ty+33,11,'#dddddd')
        text(s['mode'].capitalize()+' K',tx+12,ty+57,11)
