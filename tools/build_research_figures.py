"""Build bilingual schematic figures; no medical images or model execution.

All geometry illustrations are explanatory drawings, not predictions. Recipe
numbers are targets of ptrn-step2000-unified-v1; the development metrics are the
same rounded aggregate means documented in README, not a new evaluation.
"""
from __future__ import annotations
import argparse
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL, CORAL = '#172b35', '#657781', '#087e8b', '#d56a46'
BG, LINE, PALE, PEACH = '#faf9f6', '#dbe1df', '#e2eeeb', '#f2dfd4'


class Figure:
    def __init__(self, height, title, desc, zh=False, width=1500):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            f'<defs><style>text{{font-family:"DejaVu Sans","Noto Sans CJK SC",sans-serif;fill:{INK}}}.small{{fill:{MUTED}}}</style>',
            f'<pattern id="hatch" width="12" height="12" patternUnits="userSpaceOnUse"><rect width="12" height="12" fill="{BG}"/><path d="M-3 3L3-3M0 12L12 0M9 15L15 9" stroke="{CORAL}" stroke-width="1.3"/></pattern>',
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="{MUTED}"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="{BG}"/>']
        if zh:
            self.parts[2] = self.parts[2].replace('"DejaVu Sans","Noto Sans CJK SC"','"Noto Sans CJK SC","DejaVu Sans"')

    def raw(self, text): self.parts.append(text)
    def text(self, x, y, value, size=23, weight=400, color=INK, anchor='start'):
        self.raw(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" style="fill:{color}">{escape(value)}</text>')
    def lines(self, x, y, values, size=23, color=MUTED, step=32):
        for i, v in enumerate(values): self.text(x, y+i*step, v, size, color=color)
    def line(self, x1,y1,x2,y2,color=LINE,width=2,dash='',arrow=False):
        more = (f' stroke-dasharray="{dash}"' if dash else '') + (' marker-end="url(#arrow)"' if arrow else '')
        self.raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{more}/>')
    def rect(self, x,y,w,h,fill=BG,stroke=LINE,sw=2):
        self.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def poly(self, points,fill=PALE,stroke=TEAL,sw=3,dash=''):
        pts=' '.join(f'{x:g},{y:g}' for x,y in points)
        self.raw(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-dasharray="{dash}"/>')
    def dot(self,x,y,r=6,fill=CORAL):
        self.raw(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')
    def header(self,number,title,subtitle):
        self.text(48,48,number,18,700,TEAL)
        self.text(48,98,title,36,700)
        self.text(48,136,subtitle,22,color=MUTED)
        self.line(48,162,self.width-48,162)
    def save(self,path):
        raw='\n'.join(self.parts+['</svg>'])+'\n'
        ET.fromstring(raw)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(raw)


def lerp(a,b,t): return (a[0]*(1-t)+b[0]*t,a[1]*(1-t)+b[1]*t)


def square_map(q):
    """Unit square to quadrilateral homography for the conceptual lattice."""
    (x0,y0),(x1,y1),(x2,y2),(x3,y3)=q
    dx1,dx2,dx3=x1-x2,x3-x2,x0-x1+x2-x3
    dy1,dy2,dy3=y1-y2,y3-y2,y0-y1+y2-y3
    det=dx1*dy2-dx2*dy1
    g,h=(dx3*dy2-dx2*dy3)/det,(dx1*dy3-dx3*dy1)/det
    a,b=x1-x0+g*x1,x3-x0+h*x3
    d,e=y1-y0+g*y1,y3-y0+h*y3
    def at(u,v):
        scale=g*u+h*v+1
        return ((a*u+b*v+x0)/scale,(d*u+e*v+y0)/scale)
    return at,g,h


def grid(f, q, n=7, color=TEAL, fill=PALE):
    """Projective lattice illustrating geometry, not learned features or GT."""
    project,_,_=square_map(q)
    f.poly(q,fill,color)
    for k in range(1,n):
        a,b=project(0,k/n),project(1,k/n)
        f.line(*a,*b,color,1)
        a,b=project(k/n,0),project(k/n,1)
        f.line(*a,*b,color,1)


def geometry(zh):
    choose=lambda en,cn: cn if zh else en
    title=choose('Rectify the captured part. Keep missing pixels missing.','摆正已拍到的部分，保留未拍到区域的空缺')
    f=Figure(780,title,'Conceptual geometry diagram, not an image or measured model output. A full plane can extend beyond the photo; only observed support is sampled. Full-plane aspect ratio is distinct from the visible bounding canvas.',zh)
    f.header(choose('02 / GEOMETRY','02 / 几何'),title,choose('Full-plane geometry and observed image content are different things.','完整参考平面的几何，与照片中实际存在的影像，是两件事。'))
    f.text(58,210,choose('A  /  Photograph boundary','A  /  照片边界'),25,700)
    f.text(560,210,choose('B  /  Visible intersection','B  /  可见交集'),25,700)
    f.text(1050,210,choose('C  /  Rectified canvas','C  /  摆正画布'),25,700)
    q=[(100,230),(440,320),(400,540),(124,510)]
    f.rect(64,284,392,270,'#edece7',MUTED)
    grid(f,q)
    # Overlay the part outside the photographed rectangle with a visibly hatched region.
    t=(284-230)/(320-230)
    u=(284-230)/(510-230)
    top=[q[0],lerp(q[0],q[1],t),lerp(q[0],q[3],u)]
    f.poly(top,'url(#hatch)',CORAL,2,'6 5')
    f.line(64,284,456,284,INK,3)
    for i,(x,y) in enumerate(q):
        f.dot(x,y,6,TEAL); f.text(x+10,y+25 if i==0 else y-12,str(i+1),19,700,TEAL)
    f.text(80,595,choose('Dashed corner: outside photo','虚线角区：位于照片之外'),21,color=MUTED)
    f.line(483,403,527,403,MUTED,2,arrow=True)
    q2=[(x+501,y) for x,y in q]
    # An abstract support diagram shows the same partial plane, clipped to the photo.
    f.raw('<defs><clipPath id="photo"><rect x="565" y="284" width="392" height="270"/></clipPath></defs>')
    f.rect(565,284,392,270,'#edece7',MUTED)
    f.raw('<g clip-path="url(#photo)">'); grid(f,q2); f.raw('</g>')
    f.text(576,595,choose('Only this support has RGB values','只有可见交集内存在照片像素'),21,color=MUTED)
    f.line(984,403,1028,403,MUTED,2,arrow=True)
    q3=[(1070,284),(1430,284),(1430,554),(1070,554)]
    grid(f,q3)
    _,g,h=square_map(q)
    unit_u=t/(1+g*(1-t));unit_v=u/(1+h*(1-u))
    f.poly([(1070,284),(1070+360*unit_u,284),(1070,284+270*unit_v)],'url(#hatch)',CORAL,2)
    f.line(1070,256,1430,256,MUTED,1.5)
    f.text(1250,243,'W',20,500,MUTED,'middle')
    f.line(1451,284,1451,554,MUTED,1.5)
    f.text(1466,427,'H',20,500,MUTED)
    f.text(1070,595,choose('Full-plane ratio r = W / H','完整平面比例 r = W / H'),21,color=MUTED)
    f.line(48,631,1452,631)
    f.dot(63,669,6,TEAL);f.text(80,677,choose('Observed image region','实际拍到的影像'),21)
    f.rect(385,657,24,24,'url(#hatch)',CORAL,1)
    f.text(423,677,choose('Unobserved / invalid','未拍到／无效区域'),21)
    f.text(775,677,choose('Original RGB → one sampling pass → output + valid mask','原照片 RGB → 一次采样 → 输出图与有效掩码'),21,500)
    f.text(48,735,choose('Schematic only. The visible output bounding box need not have the full-plane aspect ratio.','仅为几何示意。可见部分的输出包围框，宽高比不一定等于完整参考平面的比例。'),20,color=MUTED)
    return f


def synthesis(zh):
    c=lambda en,cn:cn if zh else en
    title=c('A photograph is more than a warped rectangle','一张拍摄照片，不只是一个透视四边形')
    f=Figure(760,title,'Schematic of separate source reference, projective geometry, background context, and capture appearance. Drawings are not training examples. Labels keep geometry separate from appearance.',zh)
    f.header(c('03 / SYNTHESIS','03 / 数据合成'),title,c('Sample geometry, surroundings and capture artifacts separately.','参考平面、相机几何、周围背景与拍摄外观，分别设置。'))
    xvals=[56,420,784,1148]
    headings=[c('Reference','定义参考'),c('Project + crop','投影与裁损'),c('Add surroundings','合成背景'),c('Capture appearance','拍摄外观')]
    for i,x in enumerate(xvals):
        f.text(x,213,f'0{i+1}',22,700,TEAL); f.text(x+46,213,headings[i],25,700)
        if i<3:f.line(x+308,365,x+338,365,MUTED,2,arrow=True)
    grid(f,[(85,262),(290,262),(290,450),(85,450)])
    f.rect(110,282,160,150,'none',CORAL,2)
    f.lines(56,500,[c('Public-source radiograph','公开来源的 X 光片'),c('Pre-crop → new reference','母片预裁剪 → 新参考平面')],21)
    q=[(457,258),(704,310),(653,459),(435,410)]
    grid(f,q)
    f.rect(434,290,270,180,'none',INK,2)
    f.line(434,290,704,290,CORAL,2,'7 5')
    f.lines(420,500,[c('Rotation + perspective','旋转＋透视'),c('Photo crop → missing content','拍摄后裁剪 → 区域未入镜')],21)
    f.rect(784,290,270,180,'#d5d4c9',MUTED,1)
    f.rect(784,290,270,18,INK,INK,1)
    f.raw('<defs><clipPath id="composed"><rect x="784" y="290" width="270" height="180"/></clipPath></defs>')
    f.raw('<g clip-path="url(#composed)">')
    grid(f,[(x+350,y) for x,y in q],7)
    f.raw('</g>')
    f.lines(784,500,[c('Viewer / COCO context','查看器／COCO 场景'),c('Near-color / mixed-color','近色／混色背景')],21)
    f.rect(1148,290,270,180,'#d5d4c9',MUTED,1)
    f.rect(1148,290,270,18,INK,INK,1)
    f.raw('<defs><clipPath id="captured"><rect x="1148" y="290" width="270" height="180"/></clipPath></defs>')
    f.raw('<g clip-path="url(#captured)">')
    grid(f,[(x+714,y) for x,y in q],7)
    for k in range(14): f.line(1150,293+13*k,1416,296+13*k,MUTED,.7)
    f.raw('<ellipse cx="1340" cy="321" rx="60" ry="12" transform="rotate(-30 1340 321)" fill="#fff" opacity=".65"/>')
    f.raw('</g>')
    f.lines(1148,500,[c('Screen or physical film','屏幕或实体胶片'),c('Reflection / blur / interference','反光／模糊／拍摄干扰')],21)
    f.line(48,565,1452,565)
    f.text(56,614,c('Geometry labels travel with the scene','几何标签随场景变换'),26,700)
    f.text(56,655,c('Ordered corners · full-plane ratio · visible support','有序四角 · 完整平面比例 · 可见区域'),22,color=TEAL)
    f.text(790,614,c('Appearance does not redefine geometry','外观干扰不改写几何真值'),26,700)
    f.text(790,655,c('A reflection is not a missing film region.','反光遮住的区域，不冒充照片缺失区域。'),22,color=MUTED)
    f.text(56,714,c('Illustrations of the process, not generated radiographs. Fixed development scenes keep the earlier recipe.','此图为合成过程示意，不是实际生成的 X 光片。固定开发场景仍沿用早期配方。'),20,color=MUTED)
    return f


def recipe(zh):
    c=lambda en,cn:cn if zh else en
    title=c('Mostly large, complete films. Some difficult edges.','大部分片子占幅较大、内容完整，同时保留困难边缘')
    f=Figure(780,title,'Targets only: 20% trusted-original slots; photo occupancy bins 20/30/35/10/5%; missing-area bins 80/12/6/2%; backgrounds 40/25/25/10%. Realized sampling can differ.',zh)
    f.header(c('04 / TRAINING RECIPE','04 / 训练配方'),title,c('Targets for the new branch; realized scene frequencies require an audit.','新分支的配方目标；实际采样频率须以场景审计为准。'))
    f.text(50,219,c('Film occupancy within photograph','片区占照片面积'),25,700)
    bins=['90–98%','75–90%','50–75%','30–50%','10–30%']; probs=[20,30,35,10,5]
    sizes=[.95,.825,.625,.4,.2]
    from math import sqrt
    for j,(label,p,s) in enumerate(zip(bins,probs,sizes)):
        x=54+j*181; y=247
        f.rect(x,y,139,93,'#ecebe5',LINE,1)
        scale=sqrt(s); w,h=139*scale,93*scale
        f.rect(x+(139-w)/2,y+(93-h)/2,w,h,PALE,TEAL,1)
        f.text(x+69,371,label,21,anchor='middle')
        f.text(x+69,409,f'{p}%',27,700,TEAL,'middle')
    f.text(1040,234,'20%',58,700,TEAL)
    f.text(1040,273,c('original keep slots','原片保留（占全部槽位）'),22)
    f.text(1040,330,'80%',58,700,INK)
    f.text(1040,369,c('photographed slots','拍摄视图（占全部槽位）'),22)
    f.line(48,445,1452,445)
    groups=[(54,c('Missing full-plane area','完整平面的缺失面积'),[80,12,6,2],[c('None','无缺损'),'1–3%','>3–8%','>8–15%']),
            (799,c('Background family','背景家族'),[40,25,25,10],[c('Viewer','查看器'),'COCO',c('Near-color','近色'),c('Mixed','混色')])]
    colors=[TEAL,'#60a5a9','#a9c7c2',CORAL]
    for x,heading,values,labels in groups:
        f.text(x,496,heading,25,700); pos=x
        for p,col in zip(values,colors):
            w=646*p/100;f.rect(pos,523,w,25,col,col,0);pos+=w
        for j,(p,lab,col) in enumerate(zip(values,labels,colors)):
            xx=x+(j%2)*330; yy=591+(j//2)*40
            f.dot(xx+6,yy-7,6,col);f.text(xx+23,yy,f'{lab}  ·  {p}%',21)
    f.text(54,706,c('All occupancy, missing-area and background percentages are conditional on a photographed slot.','占幅、缺损及背景各档概率，均以当前槽位是拍摄图为条件。'),20,color=MUTED)
    f.text(54,740,c('Source pre-crop: 12.5% of photo slots, with 5–10% target source-area loss.','母片预裁剪：拍摄槽位的 12.5%，目标裁去母片面积的 5–10%。'),20,color=MUTED)
    return f


def tradeoff(zh):
    c=lambda en,cn:cn if zh else en
    title=c('More visible film, a little more background','少切掉一些片区，同时多留一些背景')
    f=Figure(600,title,'Same 4589 exposed synthetic photo views: mean additional crop 3.43% to2.25%; mean extra background2.54% to3.54%. Lower better in both; no independent real validation.',zh)
    f.header(c('05 / DEVELOPMENT EVIDENCE','05 / 开发集结果'),title,c('Same 4,589 exposed synthetic photographs · means · lower is better for both metrics','同一批 4,589 个已暴露合成拍摄视图 · 均值 · 两项均越低越好'))
    for x,heading,vals,delta in [(56,c('Additional crop of visible film','可见片区的新增裁损'),[3.43,2.25],'−1.18 pp'),(795,c('Extra background','预测区域的残留背景'),[2.54,3.54],'+1.00 pp')]:
        f.text(x,219,heading,27,700)
        left=x+141; right=x+611
        for k in range(5):
            xx=left+k*(right-left)/4
            f.line(xx,264,xx,404,LINE,1); f.text(xx,437,f'{k}%',20,color=MUTED,anchor='middle')
        for j,(v,col) in enumerate(zip(vals,[MUTED,CORAL])):
            yy=290+j*95
            f.text(x,yy+7,'+2000' if j==0 else '+2250',24,700,col)
            xx=left+v/4*(right-left)
            f.line(left,yy,xx,yy,col,4)
            f.dot(xx,yy,8,col)
            f.text(xx,yy-22,f'{v:.2f}%',25,700,col,'middle')
        f.text(x,493,delta.replace('pp','个百分点') if zh else delta,33,700,TEAL if delta.startswith('−') else CORAL)
    f.line(48,522,1452,522)
    f.text(56,566,c('Descriptive checkpoint comparison. Not independent real-photo accuracy or clinical validation.','这是检查点的描述性对照，不代表独立实拍准确率或临床验证。'),22,color=MUTED)
    return f


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'docs/figures');args=p.parse_args()
    for name,fn in [('geometry',geometry),('synthesis',synthesis),('recipe',recipe),('tradeoff',tradeoff)]:
        for zh in (False,True):
            target=args.output/('i18n/'+name+'.zh-CN.svg' if zh else name+'.svg')
            fn(zh).save(target); print(target)

if __name__=='__main__':main()
