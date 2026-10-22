"""Python-rendered launcher theme inspired by the user's UI design references.

Original implementation: no React, remote assets, web server or screenshot API.
"""
from functools import lru_cache
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from crowd_detector import thermal_image

BG='#090909';CARD='#111111';BORDER='#2b2b2b';TEXT='#f2f2f2';MUTED='#919191'
ACCENT='#ededed';FILL='#e5e5e5';SOFT='#292929';GREEN='#e0e0e0'

@lru_cache(maxsize=32)
def font(size=14,bold=False):
    candidates=[('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf'),
                ('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf'),
                ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    for name in candidates:
        if Path(name).is_file():return ImageFont.truetype(name,size)
    return ImageFont.load_default(size=size)


def thermal_view(ui,data,row,width):
    factor=width/160
    if ui.palette==0:
        # Exactly the annotation-stage display stretch. It never feeds inference.
        from dataset_pipeline import colorize_view
        im=colorize_view(data,ui.contrast)
    else:
        grey=thermal_image(data)[:,:,0]
        im=cv2.applyColorMap(grey,cv2.COLORMAP_VIRIDIS) if ui.palette==1 else cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
    im=cv2.resize(im,(width,round(width*.75)),interpolation=cv2.INTER_NEAREST)
    def box(b,col,label=None):
        x,y,w,h=b;p=(round(x*factor),round(y*factor))
        cv2.rectangle(im,p,(round((x+w)*factor),round((y+h)*factor)),col,1)
        if label:cv2.putText(im,label,(max(0,min(p[0],width-80)),max(14,p[1]-4)),0,.4,col,1,cv2.LINE_AA)
    if ui.raw:
        for d in row['base_detections']:box(d['bbox'],(240,180,70))
    if ui.boxes:
        for t in row['observed']:
            box(t['bbox'],(110,230,135) if t['source']=='base' else (50,210,255),
                f'#{t["id"]} {t["conf"]:.2f}' if ui.labels else None)
    if ui.show_search and row['mode']=='faster':
        for r in row['regions']:
            center=(round(r['cx']*factor),round(r['cy']*factor))
            cv2.circle(im,center,round(r['radius']*factor),(220,175,100),1)
            cv2.putText(im,f'search #{r["id"]}',center,0,.32,(220,175,100),1,cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))


def render_dashboard(ui,data,row,scale):
    W=max(1120,min(1440,160*scale+440));H=max(840,min(960,660+36*scale))
    surface=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(surface)
    ui.buttons=[];ui.sliders={}
    def text(label,x,y,size=14,color=TEXT,bold=False):
        d.text((x,y),str(label),font=font(size,bold),fill=color)
    def card(x,y,w,h):d.rounded_rectangle((x,y,x+w,y+h),radius=16,fill=CARD,outline=BORDER)
    def fit(label,width,size=13):
        f=font(size)
        if d.textlength(label,font=f)<=width:return label
        while label and d.textlength(label+'…',font=f)>width:label=label[:-1]
        return label+'…'
    def badge(label,x,y,color=ACCENT,fill=SOFT):
        w=round(d.textlength(label,font=font(11,True)))+20
        d.rounded_rectangle((x,y,x+w,y+24),radius=8,fill=fill)
        text(label,x+10,y+5,11,color,True)
        return w
    def button(label,key,x,y,w,h=34,active=False,primary=False,danger=False):
        hovered=ui.pointer is not None and x<=ui.pointer[0]<=x+w and y<=ui.pointer[1]<=y+h
        fill=('#e5e5e5' if danger else FILL if primary else SOFT if active else '#292929' if hovered else '#191919')
        outline=ACCENT if active else '#f2f2f2' if primary else '#444444' if hovered else BORDER
        d.rounded_rectangle((x,y,x+w,y+h),radius=9,fill=fill,outline=outline)
        text(label,x+11,y+(h-16)//2,13,'#111111' if primary or danger else TEXT,primary)
        # Separate keycaps make shortcuts visible without dominating labels.
        shortcut='SPACE' if key==' ' else key.upper()
        kw=max(20,round(d.textlength(shortcut,font=font(9)))+10)
        d.rounded_rectangle((x+w-kw-7,y+7,x+w-7,y+h-7),radius=4,fill='#c9c9c9' if primary or danger else '#303030')
        text(shortcut,x+w-kw-2,y+11,9,'#222222' if primary or danger else '#dddddd')
        ui.buttons.append(((x,y,x+w,y+h),key))
    # Header with a single, useful product identity.
    d.rounded_rectangle((24,20,62,58),radius=12,fill=FILL)
    text('F',36,26,23,BG,True)
    text('FLUXNET',75,21,19,TEXT,True);text('Occupancy / Thermal workspace',75,46,12,MUTED)
    button('History','o',W-494,24,100,active=ui.show_history)
    badge('EXPERIMENTAL',W-382,28)
    badge('PAUSED' if ui.paused else 'LIVE' if ui.live else 'REPLAY',W-251,28,GREEN,'#242424')
    button('Quit','q',W-122,24,98)
    d.line((24,76,W-24,76),fill=BORDER)
    left=24;lw=W-408;right=W-360;rw=336
    ui.chart_bounds=None;ui.chart_hover=None
    if ui.show_history:
        from occupancy_chart import draw_history
        draw_history(surface,ui,row,(left,92,lw,192))
        thermal_top=298
    else:
        # Numbers reflect actual row data, never demonstration values.
        metricw=(lw-24)//3
        for i,(label,value,hint) in enumerate([
            ('Observed',str(len(row['observed'])),'Current tracked detections'),
            ('Unresolved',str(row['unresolved']),'Not added to crowd count'),
            ('Processing',f'{row["processing_ms"]:.0f} ms','Detection + tracking')]):
            x=left+i*(metricw+12);card(x,92,metricw,82)
            text(label,x+16,104,12,MUTED);text(value,x+16,122,25,TEXT,True)
            text(hint,x+16,153,10,MUTED)
        thermal_top=188
    # Thermal view stays separate from interface chrome and generated labels.
    bottom=H-106;card(left,thermal_top,lw,bottom-thermal_top)
    text('Thermal view',left+18,thermal_top+16,15,TEXT,True)
    from dataset_pipeline import VIEWS
    palette_name=['Inferno','Viridis','Grayscale'][ui.palette]
    view_label=VIEWS[ui.contrast][0] if ui.palette==0 else 'fixed 15–45°C'
    text(f'160 × 120  /  {palette_name} · {view_label}',left+18,thermal_top+39,11,MUTED)
    badge(row['mode'].capitalize()+' K',left+lw-116,thermal_top+15)
    aw=lw-32;ah=bottom-thermal_top-70
    iw=min(aw,int(ah*4/3));ih=round(iw*.75)
    ix=left+(lw-iw)//2;iy=thermal_top+62+(ah-ih)//2
    im=thermal_view(ui,data,row,iw)
    mask=Image.new('L',im.size,0);ImageDraw.Draw(mask).rounded_rectangle((0,0,iw-1,ih-1),radius=10,fill=255)
    surface.paste(im,(ix,iy),mask)
    # Playback strip keeps the high-frequency actions under the picture.
    py=H-92;card(left,py,lw,58)
    button('Resume' if ui.paused else 'Pause',' ',left+12,py+12,140,primary=True)
    button('Step','n',left+162,py+12,87)
    button('Reset','x',left+259,py+12,98)
    button('Help','h',left+367,py+12,90,active=ui.help)
    text(f'FRAME {row["frame"]:06d}',left+lw-144,py+14,12,MUTED)
    text(f'{row["mode"].capitalize()} K'+(f' · estimate {row["estimate"]}' if row['mode']=='precise' else ''),left+lw-144,py+32,10,MUTED)
    # Confidence card with rounded rails and transient value bubbles.
    card(right,92,rw,212);text('Detection settings',right+16,108,15,TEXT,True)
    def slider(name,label,value,y,keys):
        value=ui.drag_value if ui.dragging==name else value
        text(label,right+16,y,12,MUTED);text(f'{value:.3f}',right+rw-62,y,13,TEXT,True)
        x0=right+24;x1=right+rw-24;cy=y+49
        ui.sliders[name]=dict(bounds=(x0-10,cy-16,x1+10,cy+16),rail=(x0,x1,cy))
        d.rounded_rectangle((x0,cy-4,x1,cy+4),radius=4,fill='#2b2b2b')
        cx=round(x0+(value-.01)/.98*(x1-x0))
        d.rounded_rectangle((x0,cy-4,max(x0+4,cx),cy+4),radius=4,fill=FILL)
        held=ui.dragging==name
        if held:d.ellipse((cx-15,cy-15,cx+15,cy+15),fill='#505050')
        radius=8 if held else 10
        d.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),fill=TEXT,outline=BG,width=2)
        if held or (ui.bubble_name==name and time.monotonic()<ui.bubble_until):
            bx=max(x0,min(x1-56,cx-28));by=cy-31
            d.rounded_rectangle((bx,by,bx+56,by+24),radius=8,fill='#333333',outline='#666666')
            text(f'{value:.3f}',bx+10,by+5,12,TEXT,True)
        text(keys,right+16,y+66,10,MUTED)
    slider('conf','Detection confidence',ui.session.detector.conf,132,'[  decrease     ]  increase')
    slider('low_conf','Local search confidence',ui.session.detector.low_conf,216,';  decrease     \'  increase   ·   inside search circles only')
    card(right,318,rw,116);text('Tracking',right+16,333,15,TEXT,True)
    text('Choose a policy. Auto adapts to crowding.',right+16,357,11,MUTED)
    for i,(label,key,mode) in enumerate([('Auto','1','auto'),('Precise','2','precise'),('Faster','3','faster')]):
        button(label,key,right+16+i*102,382,96,active=ui.session.mode==mode)
    card(right,448,rw,166);text('Dataset recording',right+16,464,15,TEXT,True)
    badge(str(ui.recorder.total)+' saved',right+rw-100,459,GREEN,'#242424')
    recording=bool(ui.recorder.continuous)
    button('Stop recording' if recording else 'Start recording','r',right+16,496,rw-32,primary=not recording,danger=recording)
    button('Save single frame','s',right+16,540,rw-32)
    text('Thermal + YOLO labels  ·  original capture format',right+16,590,11,MUTED)
    card(right,628,rw,H-652);text('Display',right+16,644,15,TEXT,True)
    for i,(label,key,active) in enumerate([('Search circles','c',ui.show_search),('Boxes','b',ui.boxes),('IDs / scores','i',ui.labels),('Raw detections','g',ui.raw),('Palette','p',False),('Contrast','v',ui.palette==0)]):
        x=right+16+(i%2)*156;y=674+(i//2)*44
        button(label,key,x,y,148,active=active)
    text(fit(ui.message,W-56),28,H-21,12,'#bbbbbb')
    if ui.help:
        hx=left+28;hy=276;hw=lw-56
        d.rounded_rectangle((hx,hy,hx+hw,hy+258),radius=14,fill='#181818',outline='#555555',width=1)
        text('Keyboard shortcuts',hx+20,hy+20,19,TEXT,True)
        for i,line in enumerate(['[ / ]  Detection confidence      ; / \'  Search confidence',
            '1 / 2 / 3  Auto / Precise / Faster      T  Cycle tracker',
            'R  Record / stop      S  Save one frame      X  Reset tracker',
            'Space  Pause / resume      N  Step      C  Search circles',
            'B  Boxes      I  IDs      G  Raw detections      P  Palette',
            'V  Contrast      O  History chart      H  Close help      Q  Quit']):text(line,hx+20,hy+62+i*27,12,'#cccccc')
        text('Confidence changes take effect on the next processed frame.',hx+20,hy+232,11,MUTED)
    return cv2.cvtColor(np.asarray(surface),cv2.COLOR_RGB2BGR)
