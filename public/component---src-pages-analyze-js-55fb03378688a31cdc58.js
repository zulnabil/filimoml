(window.webpackJsonp=window.webpackJsonp||[]).push([[5],{"1Yqg":function(t,e,r){!function(e,n){if(t.exports)t.exports=n(r("SnCn"));else{var i=e.Zdog;i.RoundedRect=n(i.Shape)}}(this,(function(t){var e=t.subclass({width:1,height:1,cornerRadius:.25,closed:!1});return e.prototype.setPath=function(){var t=this.width/2,e=this.height/2,r=Math.min(t,e),n=Math.min(this.cornerRadius,r),i=t-n,o=e-n,a=[{x:i,y:-e},{arc:[{x:t,y:-e},{x:t,y:-o}]}];o&&a.push({x:t,y:o}),a.push({arc:[{x:t,y:e},{x:i,y:e}]}),i&&a.push({x:-i,y:e}),a.push({arc:[{x:-t,y:e},{x:-t,y:o}]}),o&&a.push({x:-t,y:-o}),a.push({arc:[{x:-t,y:-e},{x:-i,y:-e}]}),i&&a.push({x:i,y:-e}),this.path=a},e}))},"5y3B":function(t,e,r){var n,i,o,a,s,h,c,p,u,d,l,f,g,m,y,v,x,w,b,S,C,E;a=this,t.exports?t.exports=(s=r("hmEX"),h=r("6ZIi"),c=r("v+tz"),p=r("RtnT"),u=r("X9wK"),d=r("AW3U"),l=r("CnjW"),f=r("Y0B8"),g=r("SnCn"),m=r("9kwu"),y=r("U14w"),v=r("1Yqg"),x=r("oqrJ"),w=r("lP2j"),b=r("CXVM"),S=r("DMFv"),C=r("EWVJ"),E=r("uPLu"),s.CanvasRenderer=h,s.SvgRenderer=c,s.Vector=p,s.Anchor=u,s.Dragger=d,s.Illustration=l,s.PathCommand=f,s.Shape=g,s.Group=m,s.Rect=y,s.RoundedRect=v,s.Ellipse=x,s.Polygon=w,s.Hemisphere=b,s.Cylinder=S,s.Cone=C,s.Box=E,s):(i=[],n=a.Zdog,void 0===(o="function"==typeof n?n.apply(e,i):n)||(t.exports=o))},"6ZIi":function(t,e,r){var n,i;r("yyme"),r("QWBl"),r("FZtP"),n=this,i=function(){var t={isCanvas:!0,begin:function(t){t.beginPath()},move:function(t,e,r){t.moveTo(r.x,r.y)},line:function(t,e,r){t.lineTo(r.x,r.y)},bezier:function(t,e,r,n,i){t.bezierCurveTo(r.x,r.y,n.x,n.y,i.x,i.y)},closePath:function(t){t.closePath()},setPath:function(){},renderPath:function(e,r,n,i){this.begin(e,r),n.forEach((function(n){n.render(e,r,t)})),i&&this.closePath(e,r)},stroke:function(t,e,r,n,i){r&&(t.strokeStyle=n,t.lineWidth=i,t.stroke())},fill:function(t,e,r,n){r&&(t.fillStyle=n,t.fill())},end:function(){}};return t},t.exports?t.exports=i():n.Zdog.CanvasRenderer=i()},"7ib3":function(t,e,r){"use strict";var n=r("q1tI"),i=r.n(n),o=r("vOnD"),a=o.b.div.withConfig({displayName:"button-medium-orange__BtnWrapper",componentId:"y73r4u-0"})(["display:flex;border-radius:100%;width:90px;height:90px;background:",";box-shadow:",";transition:all 0.2s;"],(function(t){return t.clicked?"linear-gradient(134.06deg, #DB501F 15.56%, #BA391D 83.35%)":"linear-gradient(134.06deg, #BA391D 15.56%, #DB501F 83.35%)"}),(function(t){return t.clicked?"-9px -9px 20px rgba(81, 87, 97, 0.70), 15px 15px 26px rgba(0, 0, 0, 0.45)":"-8px -8px 18px rgba(81, 87, 97, 0.65), 14px 14px 24px rgba(0, 0, 0, 0.4)"})),s=o.b.div.withConfig({displayName:"button-medium-orange__Btn",componentId:"y73r4u-1"})(["display:flex;margin:auto;border-radius:100%;width:84px;height:84px;background:",";transition:all 0.2s;"],(function(t){return t.clicked?"linear-gradient(135.09deg, #BB241B 1.07%, #E65721 83.33%)":"linear-gradient(135.09deg, #E65721 17.07%, #BB241B 83.33%)"}));e.a=function(t){var e=t.children,r=t.clicked,n=t.onClick;return i.a.createElement(a,{clicked:r,onClick:n},i.a.createElement(s,{clicked:r},e))}},"9kwu":function(t,e,r){r("QWBl"),r("ToJy"),r("FZtP"),function(e,n){if(t.exports)t.exports=n(r("X9wK"));else{var i=e.Zdog;i.Group=n(i.Anchor)}}(this,(function(t){var e=t.subclass({updateSort:!1,visible:!0});return e.prototype.updateSortValue=function(){var e=0;this.flatGraph.forEach((function(t){t.updateSortValue(),e+=t.sortValue})),this.sortValue=e/this.flatGraph.length,this.updateSort&&this.flatGraph.sort(t.shapeSorter)},e.prototype.render=function(t,e){this.visible&&this.flatGraph.forEach((function(r){r.render(t,e)}))},e.prototype.updateFlatGraph=function(){this.flatGraph=this.addChildFlatGraph([])},e.prototype.getFlatGraph=function(){return[this]},e}))},AW3U:function(t,e,r){var n,i;n=this,i=function(){var t="undefined"!=typeof window,e="mousedown",r="mousemove",n="mouseup";function i(){}function o(t){this.create(t||{})}return t&&(window.PointerEvent?(e="pointerdown",r="pointermove",n="pointerup"):"ontouchstart"in window&&(e="touchstart",r="touchmove",n="touchend")),o.prototype.create=function(t){this.onDragStart=t.onDragStart||i,this.onDragMove=t.onDragMove||i,this.onDragEnd=t.onDragEnd||i,this.bindDrag(t.startElement)},o.prototype.bindDrag=function(t){(t=this.getQueryElement(t))&&(t.style.touchAction="none",t.addEventListener(e,this))},o.prototype.getQueryElement=function(t){return"string"==typeof t&&(t=document.querySelector(t)),t},o.prototype.handleEvent=function(t){var e=this["on"+t.type];e&&e.call(this,t)},o.prototype.onmousedown=o.prototype.onpointerdown=function(t){this.dragStart(t,t)},o.prototype.ontouchstart=function(t){this.dragStart(t,t.changedTouches[0])},o.prototype.dragStart=function(e,i){e.preventDefault(),this.dragStartX=i.pageX,this.dragStartY=i.pageY,t&&(window.addEventListener(r,this),window.addEventListener(n,this)),this.onDragStart(i)},o.prototype.ontouchmove=function(t){this.dragMove(t,t.changedTouches[0])},o.prototype.onmousemove=o.prototype.onpointermove=function(t){this.dragMove(t,t)},o.prototype.dragMove=function(t,e){t.preventDefault();var r=e.pageX-this.dragStartX,n=e.pageY-this.dragStartY;this.onDragMove(e,r,n)},o.prototype.onmouseup=o.prototype.onpointerup=o.prototype.ontouchend=o.prototype.dragEnd=function(){window.removeEventListener(r,this),window.removeEventListener(n,this),this.onDragEnd()},o},t.exports?t.exports=i():n.Zdog.Dragger=i()},CXVM:function(t,e,r){r("yyme"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("RtnT"),r("X9wK"),r("oqrJ"));else{var i=e.Zdog;i.Hemisphere=n(i,i.Vector,i.Anchor,i.Ellipse)}}(this,(function(t,e,r,n){var i=n.subclass({fill:!0}),o=t.TAU;i.prototype.create=function(){n.prototype.create.apply(this,arguments),this.apex=new r({addTo:this,translate:{z:this.diameter/2}}),this.renderCentroid=new e},i.prototype.updateSortValue=function(){this.renderCentroid.set(this.renderOrigin).lerp(this.apex.renderOrigin,3/8),this.sortValue=this.renderCentroid.z},i.prototype.render=function(t,e){this.renderDome(t,e),n.prototype.render.apply(this,arguments)},i.prototype.renderDome=function(t,e){if(this.visible){var r=this.getDomeRenderElement(t,e),n=Math.atan2(this.renderNormal.y,this.renderNormal.x),i=this.diameter/2*this.renderNormal.magnitude(),a=this.renderOrigin.x,s=this.renderOrigin.y;if(e.isCanvas){var h=n+o/4,c=n-o/4;t.beginPath(),t.arc(a,s,i,h,c)}else e.isSvg&&(n=(n-o/4)/o*360,this.domeSvgElement.setAttribute("d","M "+-i+",0 A "+i+","+i+" 0 0 1 "+i+",0"),this.domeSvgElement.setAttribute("transform","translate("+a+","+s+" ) rotate("+n+")"));e.stroke(t,r,this.stroke,this.color,this.getLineWidth()),e.fill(t,r,this.fill,this.color),e.end(t,r)}};return i.prototype.getDomeRenderElement=function(t,e){if(e.isSvg)return this.domeSvgElement||(this.domeSvgElement=document.createElementNS("http://www.w3.org/2000/svg","path"),this.domeSvgElement.setAttribute("stroke-linecap","round"),this.domeSvgElement.setAttribute("stroke-linejoin","round")),this.domeSvgElement},i}))},CnjW:function(t,e,r){r("wLYn"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("X9wK"),r("AW3U"));else{var i=e.Zdog;i.Illustration=n(i,i.Anchor,i.Dragger)}}(this,(function(t,e,r){function n(){}var i=t.TAU,o=e.subclass({element:void 0,centered:!0,zoom:1,dragRotate:!1,resize:!1,onPrerender:n,onDragStart:n,onDragMove:n,onDragEnd:n,onResize:n});return t.extend(o.prototype,r.prototype),o.prototype.create=function(t){e.prototype.create.call(this,t),r.prototype.create.call(this,t),this.setElement(this.element),this.setDragRotate(this.dragRotate),this.setResize(this.resize)},o.prototype.setElement=function(t){if(!(t=this.getQueryElement(t)))throw new Error("Zdog.Illustration element required. Set to "+t);var e=t.nodeName.toLowerCase();"canvas"==e?this.setCanvas(t):"svg"==e&&this.setSvg(t)},o.prototype.setSize=function(t,e){t=Math.round(t),e=Math.round(e),this.isCanvas?this.setSizeCanvas(t,e):this.isSvg&&this.setSizeSvg(t,e)},o.prototype.setResize=function(t){this.resize=t,this.resizeListener||(this.resizeListener=this.onWindowResize.bind(this)),t?(window.addEventListener("resize",this.resizeListener),this.onWindowResize()):window.removeEventListener("resize",this.resizeListener)},o.prototype.onWindowResize=function(){this.setMeasuredSize(),this.onResize(this.width,this.height)},o.prototype.setMeasuredSize=function(){var t,e;if("fullscreen"==this.resize)t=window.innerWidth,e=window.innerHeight;else{var r=this.element.getBoundingClientRect();t=r.width,e=r.height}this.setSize(t,e)},o.prototype.renderGraph=function(t){this.isCanvas?this.renderGraphCanvas(t):this.isSvg&&this.renderGraphSvg(t)},o.prototype.updateRenderGraph=function(t){this.updateGraph(),this.renderGraph(t)},o.prototype.setCanvas=function(t){this.element=t,this.isCanvas=!0,this.ctx=this.element.getContext("2d"),this.setSizeCanvas(t.width,t.height)},o.prototype.setSizeCanvas=function(t,e){this.width=t,this.height=e;var r=this.pixelRatio=window.devicePixelRatio||1;this.element.width=this.canvasWidth=t*r,this.element.height=this.canvasHeight=e*r,r>1&&!this.resize&&(this.element.style.width=t+"px",this.element.style.height=e+"px")},o.prototype.renderGraphCanvas=function(t){t=t||this,this.prerenderCanvas(),e.prototype.renderGraphCanvas.call(t,this.ctx),this.postrenderCanvas()},o.prototype.prerenderCanvas=function(){var t=this.ctx;if(t.lineCap="round",t.lineJoin="round",t.clearRect(0,0,this.canvasWidth,this.canvasHeight),t.save(),this.centered){var e=this.width/2*this.pixelRatio,r=this.height/2*this.pixelRatio;t.translate(e,r)}var n=this.pixelRatio*this.zoom;t.scale(n,n),this.onPrerender(t)},o.prototype.postrenderCanvas=function(){this.ctx.restore()},o.prototype.setSvg=function(t){this.element=t,this.isSvg=!0,this.pixelRatio=1;var e=t.getAttribute("width"),r=t.getAttribute("height");this.setSizeSvg(e,r)},o.prototype.setSizeSvg=function(t,e){this.width=t,this.height=e;var r=t/this.zoom,n=e/this.zoom,i=this.centered?-r/2:0,o=this.centered?-n/2:0;this.element.setAttribute("viewBox",i+" "+o+" "+r+" "+n),this.resize?(this.element.removeAttribute("width"),this.element.removeAttribute("height")):(this.element.setAttribute("width",t),this.element.setAttribute("height",e))},o.prototype.renderGraphSvg=function(t){t=t||this,function(t){for(;t.firstChild;)t.removeChild(t.firstChild)}(this.element),this.onPrerender(this.element),e.prototype.renderGraphSvg.call(t,this.element)},o.prototype.setDragRotate=function(t){t&&(!0===t&&(t=this),this.dragRotate=t,this.bindDrag(this.element))},o.prototype.dragStart=function(){this.dragStartRX=this.dragRotate.rotate.x,this.dragStartRY=this.dragRotate.rotate.y,r.prototype.dragStart.apply(this,arguments)},o.prototype.dragMove=function(t,e){var n=e.pageX-this.dragStartX,o=e.pageY-this.dragStartY,a=Math.min(this.width,this.height),s=n/a*i,h=o/a*i;this.dragRotate.rotate.x=this.dragStartRX-h,this.dragRotate.rotate.y=this.dragStartRY-s,r.prototype.dragMove.apply(this,arguments)},o}))},DMFv:function(t,e,r){r("yyme"),r("QWBl"),r("eoL8"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("Y0B8"),r("SnCn"),r("9kwu"),r("oqrJ"));else{var i=e.Zdog;i.Cylinder=n(i,i.PathCommand,i.Shape,i.Group,i.Ellipse)}}(this,(function(t,e,r,n,i){function o(){}var a=n.subclass({color:"#333",updateSort:!0});a.prototype.create=function(){n.prototype.create.apply(this,arguments),this.pathCommands=[new e("move",[{}]),new e("line",[{}])]},a.prototype.render=function(t,e){this.renderCylinderSurface(t,e),n.prototype.render.apply(this,arguments)},a.prototype.renderCylinderSurface=function(t,e){if(this.visible){var r=this.getRenderElement(t,e),n=this.frontBase,i=this.rearBase,o=n.renderNormal.magnitude(),a=n.diameter*o+n.getLineWidth();this.pathCommands[0].renderPoints[0].set(n.renderOrigin),this.pathCommands[1].renderPoints[0].set(i.renderOrigin),e.isCanvas&&(t.lineCap="butt"),e.renderPath(t,r,this.pathCommands),e.stroke(t,r,!0,this.color,a),e.end(t,r),e.isCanvas&&(t.lineCap="round")}};a.prototype.getRenderElement=function(t,e){if(e.isSvg)return this.svgElement||(this.svgElement=document.createElementNS("http://www.w3.org/2000/svg","path")),this.svgElement},a.prototype.copyGraph=o,i.subclass().prototype.copyGraph=o;var s=r.subclass({diameter:1,length:1,frontFace:void 0,fill:!0}),h=t.TAU;s.prototype.create=function(){r.prototype.create.apply(this,arguments),this.group=new a({addTo:this,color:this.color,visible:this.visible});var t=this.length/2,e=this.backface||!0;this.frontBase=this.group.frontBase=new i({addTo:this.group,diameter:this.diameter,translate:{z:t},rotate:{y:h/2},color:this.color,stroke:this.stroke,fill:this.fill,backface:this.frontFace||e,visible:this.visible}),this.rearBase=this.group.rearBase=this.frontBase.copy({translate:{z:-t},rotate:{y:0},backface:e})},s.prototype.render=function(){};return["stroke","fill","color","visible"].forEach((function(t){var e="_"+t;Object.defineProperty(s.prototype,t,{get:function(){return this[e]},set:function(r){this[e]=r,this.frontBase&&(this.frontBase[t]=r,this.rearBase[t]=r,this.group[t]=r)}})})),s}))},EWVJ:function(t,e,r){r("yyme"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("RtnT"),r("Y0B8"),r("X9wK"),r("oqrJ"));else{var i=e.Zdog;i.Cone=n(i,i.Vector,i.PathCommand,i.Anchor,i.Ellipse)}}(this,(function(t,e,r,n,i){var o=i.subclass({length:1,fill:!0}),a=t.TAU;o.prototype.create=function(){i.prototype.create.apply(this,arguments),this.apex=new n({addTo:this,translate:{z:this.length}}),this.renderApex=new e,this.renderCentroid=new e,this.tangentA=new e,this.tangentB=new e,this.surfacePathCommands=[new r("move",[{}]),new r("line",[{}]),new r("line",[{}])]},o.prototype.updateSortValue=function(){this.renderCentroid.set(this.renderOrigin).lerp(this.apex.renderOrigin,1/3),this.sortValue=this.renderCentroid.z},o.prototype.render=function(t,e){this.renderConeSurface(t,e),i.prototype.render.apply(this,arguments)},o.prototype.renderConeSurface=function(t,e){if(this.visible){this.renderApex.set(this.apex.renderOrigin).subtract(this.renderOrigin);var r=this.renderNormal.magnitude(),n=this.renderApex.magnitude2d(),i=this.renderNormal.magnitude2d(),o=Math.acos(i/r),s=Math.sin(o),h=this.diameter/2*r;if(h*s<n){var c=Math.atan2(this.renderNormal.y,this.renderNormal.x)+a/2,p=n/s,u=Math.acos(h/p),d=this.tangentA,l=this.tangentB;d.x=Math.cos(u)*h*s,d.y=Math.sin(u)*h,l.set(this.tangentA),l.y*=-1,d.rotateZ(c),l.rotateZ(c),d.add(this.renderOrigin),l.add(this.renderOrigin),this.setSurfaceRenderPoint(0,d),this.setSurfaceRenderPoint(1,this.apex.renderOrigin),this.setSurfaceRenderPoint(2,l);var f=this.getSurfaceRenderElement(t,e);e.renderPath(t,f,this.surfacePathCommands),e.stroke(t,f,this.stroke,this.color,this.getLineWidth()),e.fill(t,f,this.fill,this.color),e.end(t,f)}}};return o.prototype.getSurfaceRenderElement=function(t,e){if(e.isSvg)return this.surfaceSvgElement||(this.surfaceSvgElement=document.createElementNS("http://www.w3.org/2000/svg","path"),this.surfaceSvgElement.setAttribute("stroke-linecap","round"),this.surfaceSvgElement.setAttribute("stroke-linejoin","round")),this.surfaceSvgElement},o.prototype.setSurfaceRenderPoint=function(t,e){this.surfacePathCommands[t].renderPoints[0].set(e)},o}))},"JaO/":function(t,e,r){"use strict";r.r(e);r("pNMO"),r("4Brf"),r("0oug"),r("ma9I"),r("yyme"),r("pjDv"),r("yXV3"),r("4mDm"),r("2B1R"),r("E9XD"),r("+2oP"),r("sMBO"),r("07d7"),r("5s+n"),r("JfAA"),r("PKPk"),r("3bBZ");var n=r("o0o1"),i=r.n(n),o=(r("ls82"),r("q1tI")),a=r.n(o),s=(r("Wbzz"),r("vOnD")),h=(r("5y3B"),r("mrSG")),c=r("OXR1"),p=o.forwardRef((function(t,e){return o.createElement(c.a,Object(h.a)({iconAttrs:{fill:"currentColor",xmlns:"http://www.w3.org/2000/svg"},iconVerticalAlign:"middle",iconViewBox:"0 0 24 24"},t,{ref:e}),o.createElement("path",{d:"M5 11h14v2H5z",key:"k0"}))}));p.displayName="Minus";var u=o.forwardRef((function(t,e){return o.createElement(c.a,Object(h.a)({iconAttrs:{fill:"currentColor",xmlns:"http://www.w3.org/2000/svg"},iconVerticalAlign:"middle",iconViewBox:"0 0 24 24"},t,{ref:e}),o.createElement("path",{d:"M19 11h-6V5h-2v6H5v2h6v6h2v-6h6z",key:"k0"}))}));u.displayName="Plus";var d=o.forwardRef((function(t,e){return o.createElement(c.a,Object(h.a)({iconAttrs:{fill:"currentColor",xmlns:"http://www.w3.org/2000/svg"},iconVerticalAlign:"middle",iconViewBox:"0 0 24 24"},t,{ref:e}),o.createElement("path",{d:"M12 16c2.206 0 4-1.794 4-4V6c0-2.217-1.785-4.021-3.979-4.021a.933.933 0 00-.209.025A4.006 4.006 0 008 6v6c0 2.206 1.794 4 4 4z",key:"k0"}),o.createElement("path",{d:"M11 19.931V22h2v-2.069c3.939-.495 7-3.858 7-7.931h-2c0 3.309-2.691 6-6 6s-6-2.691-6-6H4c0 4.072 3.061 7.436 7 7.931z",key:"k1"}))}));d.displayName="Microphone";var l=o.forwardRef((function(t,e){return o.createElement(c.a,Object(h.a)({iconAttrs:{fill:"currentColor",xmlns:"http://www.w3.org/2000/svg"},iconVerticalAlign:"-.125em",iconViewBox:"0 0 448 512"},t,{ref:e}),o.createElement("path",{fill:"currentColor",d:"M400 32H48C21.5 32 0 53.5 0 80v352c0 26.5 21.5 48 48 48h352c26.5 0 48-21.5 48-48V80c0-26.5-21.5-48-48-48z",key:"k0"}))}));l.displayName="Stop";var f=r("Efxr"),g=r("4pX0"),m=r.n(g),y=r("Bl7J"),v=r("vrFN"),x=r("7ib3");function w(t){return function(t){if(Array.isArray(t))return b(t)}(t)||function(t){if("undefined"!=typeof Symbol&&Symbol.iterator in Object(t))return Array.from(t)}(t)||function(t,e){if(!t)return;if("string"==typeof t)return b(t,e);var r=Object.prototype.toString.call(t).slice(8,-1);"Object"===r&&t.constructor&&(r=t.constructor.name);if("Map"===r||"Set"===r)return Array.from(t);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return b(t,e)}(t)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function b(t,e){(null==e||e>t.length)&&(e=t.length);for(var r=0,n=new Array(e);r<e;r++)n[r]=t[r];return n}function S(t,e,r,n,i,o,a){try{var s=t[o](a),h=s.value}catch(c){return void r(c)}s.done?e(h):Promise.resolve(h).then(n,i)}function C(t){return function(){var e=this,r=arguments;return new Promise((function(n,i){var o=t.apply(e,r);function a(t){S(o,n,i,a,s,"next",t)}function s(t){S(o,n,i,a,s,"throw",t)}a(void 0)}))}}var E=s.b.div.withConfig({displayName:"analyze__AudioVisualizer",componentId:"sc-6g8wlf-0"})(["position:relative;display:inline-block;margin-top:100px;border-radius:100%;width:230px;height:230px;background:linear-gradient(134.06deg,#23262a 15.56%,#23262a 83.35%);box-shadow:-16px -16px 30px rgba(81,87,97,0.4),16px 16px 30px rgba(0,0,0,0.35);"]),z=s.b.div.withConfig({displayName:"analyze__ChildAudioVisualizer",componentId:"sc-6g8wlf-1"})(["position:absolute;top:50%;left:50%;transform:translateY(-50%) translateX(-50%);border-radius:100%;width:215px;height:215px;background:linear-gradient( 135.09deg,rgba(44,48,53,0.182) 40.54%,rgba(30,33,36,0.7) 83.33% );mix-blend-mode:multiply;"]),k=s.b.p.withConfig({displayName:"analyze__TxtDesc",componentId:"sc-6g8wlf-2"})(["margin:30px 15px 15px;font-size:14px;font-weight:normal;line-height:16px;text-align:center;color:rgba(255,255,255,0.5);"]),P=s.b.p.withConfig({displayName:"analyze__TxtResult",componentId:"sc-6g8wlf-3"})(["margin:15px;font-size:32px;font-weight:bold;line-height:37px;text-align:center;color:rgba(255,255,255,0.8);"]),R=s.b.div.withConfig({displayName:"analyze__ContainerVolume",componentId:"sc-6g8wlf-4"})(["width:100%;max-width:315px;margin:45px auto;display:flex;align-items:center;"]),A=s.b.input.withConfig({displayName:"analyze__SliderVolume",componentId:"sc-6g8wlf-5"})(["border:0;-webkit-appearance:none;width:100%;height:8px;background:linear-gradient( 180deg,#101010 32.29%,#23262a 77.6%,#505862 91.15% );border-radius:5px;outline:none;opacity:0.7;-webkit-transition:0.2s;transition:all 0.2s;border:0;&::-webkit-slider-runnable-track{margin-left:2px;background:"," height:4px;border-radius:5px;transition:all 0.2s;border:0;}&::-webkit-slider-thumb{-webkit-appearance:none;width:32px;height:32px;border-radius:100%;cursor:pointer;margin-top:-14px;border:0;background:linear-gradient(133.83deg,#373b42 19.39%,#202326 87.47%);box-shadow:4px 4px 12px rgba(0,0,0,0.5);}&::-moz-range-track{margin-left:5px;background:"," height:4px;border-radius:5px;border:0;}&::-moz-focus-outer{border:0;}&::-moz-range-thumb{border:0;width:32px;height:32px;border-radius:50%;margin-top:-14px;background:linear-gradient(133.83deg,#373b42 19.39%,#202326 87.47%);box-shadow:4px 4px 12px rgba(0,0,0,0.5);}"],(function(t){return"linear-gradient(90deg, #e55521 0%, #f5a42a "+(t.volume-1)+"%, rgba(0, 0, 0, 0) "+t.volume+"%);"}),(function(t){return"linear-gradient(90deg, #e55521 0%, #f5a42a "+(t.volume-1)+"%, rgba(0, 0, 0, 0) "+t.volume+"%);"})),M=Object(s.b)(p).withConfig({displayName:"analyze__IconMinus",componentId:"sc-6g8wlf-6"})(["margin-right:5px;color:white;opacity:0.65;width:12px;height:12px;"]),O=Object(s.b)(u).withConfig({displayName:"analyze__IconPlus",componentId:"sc-6g8wlf-7"})(["margin-left:5px;color:white;opacity:0.65;width:12px;height:12px;"]),B=Object(s.b)(d).withConfig({displayName:"analyze__IconMic",componentId:"sc-6g8wlf-8"})(["margin:auto;color:white;opacity:0.65;width:32px;height:32px;"]),F=Object(s.b)(l).withConfig({displayName:"analyze__IconStop",componentId:"sc-6g8wlf-9"})(["margin:auto;color:white;opacity:0.65;width:22px;height:22px;"]),G=f.a.div({visible:{y:0,opacity:1},hidden:{y:50,opacity:0},initialPose:"hidden"}),V=function(){var t=C(i.a.mark((function t(){return i.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,navigator.mediaDevices.getUserMedia({audio:!0,video:!1});case 3:return t.abrupt("return",t.sent);case 6:t.prev=6,t.t0=t.catch(0),console.log("Error:",t.t0);case 9:case"end":return t.stop()}}),t,null,[[0,6]])})));return function(){return t.apply(this,arguments)}}();function D(t){return t[0].map((function(e,r){return t.map((function(t){return t[r]})).reduce((function(t,e){return t+e}),0)/t.length}))}e.default=function(){var t=a.a.useState(50),e=t[0],r=t[1],n=a.a.useState(null),o=n[0],s=n[1],h=a.a.useState([]),c=h[0],p=h[1];function u(t){switch(t.indexOf(Math.max.apply(Math,w(t)))){case 0:return"Female Angry";case 1:return"Female Happy";case 2:return"Female Neutral";case 3:return"Female Sad";case 4:return"Male Angry";case 5:return"Male Happy";case 6:return"Male Neutral";case 7:return"Male Sad"}}a.a.useLayoutEffect((function(){o&&o.start()}),[o]);var d=a.a.useState(!1),l=d[0],f=d[1],g=a.a.useState(null),b=(g[0],g[1],a.a.useState("")),S=b[0],T=b[1],X=a.a.useState("Tekan tombol rekam untuk mengenali emosi."),N=X[0],Z=X[1],j=new Array(24).fill(0),I=a.a.useState({total:0,specific:j}),W=(I[0],I[1],a.a.useState(0)),_=(W[0],W[1],function(){var t=C(i.a.mark((function t(){var e,r,n;return i.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:f(!l),l?(o.stop(),console.log("Analyzing..."),Z("Mohon tunggu untuk sementara waktu."),T("Menganalisis suara..."),n=D(c),console.log(n),fetch("https://filimoml.com/speeches/predict",{method:"post",headers:{"Content-type":"application/json"},body:JSON.stringify({mfcc0:n[0],mfcc1:n[1],mfcc2:n[2],mfcc3:n[3],mfcc4:n[4],mfcc5:n[5],mfcc6:n[6],mfcc7:n[7],mfcc8:n[8],mfcc9:n[9],mfcc10:n[10],mfcc11:n[11],mfcc12:n[12]})}).then((function(t){return t.json()})).then((function(t){console.log(t),Z("Emosi terdeteksi"),T(u(t.output[0]))})),p([])):((e=window.AudioContext||window.webkitAudioContext||!1)||alert("Your browser not supported"),(r=new e).resume().then((function(){console.log("Playback resumed successfully")})),V().then((function(t){var e=r.createMediaStreamSource(t);s(m.a.createMeydaAnalyzer({audioContext:r,source:e,bufferSize:512,featureExtractors:["mfcc"],callback:function(t){p((function(e){return e.concat([t.mfcc])})),console.log("meyda initialized")}}))})),Z("Tekan stop untuk mulai menganalisis."),T("Merekam..."));case 2:case"end":return t.stop()}}),t)})));return function(){return t.apply(this,arguments)}}());return a.a.createElement(y.a,{havemenu:!0},a.a.createElement(v.a,{title:"Analyze"}),a.a.createElement(G,null,a.a.createElement(E,null,a.a.createElement(z,null))),a.a.createElement(G,null,a.a.createElement(k,null,N),a.a.createElement(P,null,S)),a.a.createElement(G,null,a.a.createElement(R,null,a.a.createElement(M,null),a.a.createElement(A,{onChange:function(t){r(parseFloat(t.target.value))},volume:e,type:"range",step:"1",min:"1",max:"100",value:e}),a.a.createElement(O,null))),a.a.createElement(G,null,a.a.createElement("div",{style:{display:"flex",justifyContent:"center",margin:35}},a.a.createElement(x.a,{clicked:l,onClick:_},l?a.a.createElement(F,null):a.a.createElement(B,null)))))}},RtnT:function(t,e,r){!function(e,n){if(t.exports)t.exports=n(r("hmEX"));else{var i=e.Zdog;i.Vector=n(i)}}(this,(function(t){function e(t){this.set(t)}var r=t.TAU;function n(t,e,n,i){if(e&&e%r!=0){var o=Math.cos(e),a=Math.sin(e),s=t[n],h=t[i];t[n]=s*o-h*a,t[i]=h*o+s*a}}function i(t){return Math.abs(t-1)<1e-8?1:Math.sqrt(t)}return e.prototype.set=function(t){return this.x=t&&t.x||0,this.y=t&&t.y||0,this.z=t&&t.z||0,this},e.prototype.write=function(t){return t?(this.x=null!=t.x?t.x:this.x,this.y=null!=t.y?t.y:this.y,this.z=null!=t.z?t.z:this.z,this):this},e.prototype.rotate=function(t){if(t)return this.rotateZ(t.z),this.rotateY(t.y),this.rotateX(t.x),this},e.prototype.rotateZ=function(t){n(this,t,"x","y")},e.prototype.rotateX=function(t){n(this,t,"y","z")},e.prototype.rotateY=function(t){n(this,t,"x","z")},e.prototype.isSame=function(t){return!!t&&(this.x===t.x&&this.y===t.y&&this.z===t.z)},e.prototype.add=function(t){return t?(this.x+=t.x||0,this.y+=t.y||0,this.z+=t.z||0,this):this},e.prototype.subtract=function(t){return t?(this.x-=t.x||0,this.y-=t.y||0,this.z-=t.z||0,this):this},e.prototype.multiply=function(t){return null==t||("number"==typeof t?(this.x*=t,this.y*=t,this.z*=t):(this.x*=null!=t.x?t.x:1,this.y*=null!=t.y?t.y:1,this.z*=null!=t.z?t.z:1)),this},e.prototype.transform=function(t,e,r){return this.multiply(r),this.rotate(e),this.add(t),this},e.prototype.lerp=function(e,r){return this.x=t.lerp(this.x,e.x||0,r),this.y=t.lerp(this.y,e.y||0,r),this.z=t.lerp(this.z,e.z||0,r),this},e.prototype.magnitude=function(){return i(this.x*this.x+this.y*this.y+this.z*this.z)},e.prototype.magnitude2d=function(){return i(this.x*this.x+this.y*this.y)},e.prototype.copy=function(){return new e(this)},e}))},SnCn:function(t,e,r){r("yyme"),r("QWBl"),r("yXV3"),r("J30X"),r("2B1R"),r("tkto"),r("FZtP"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("RtnT"),r("Y0B8"),r("X9wK"));else{var i=e.Zdog;i.Shape=n(i,i.Vector,i.PathCommand,i.Anchor)}}(this,(function(t,e,r,n){var i=n.subclass({stroke:1,fill:!1,color:"#333",closed:!0,visible:!0,path:[{}],front:{z:1},backface:!0});i.prototype.create=function(t){n.prototype.create.call(this,t),this.updatePath(),this.front=new e(t.front||this.front),this.renderFront=new e(this.front),this.renderNormal=new e};var o=["move","line","bezier","arc"];i.prototype.updatePath=function(){this.setPath(),this.updatePathCommands()},i.prototype.setPath=function(){},i.prototype.updatePathCommands=function(){var t;this.pathCommands=this.path.map((function(e,n){var i=Object.keys(e),a=i[0],s=e[a];1==i.length&&-1!=o.indexOf(a)||(a="line",s=e);var h="line"==a||"move"==a,c=Array.isArray(s);h&&!c&&(s=[s]);var p=new r(a=0===n?"move":a,s,t);return t=p.endRenderPoint,p}))},i.prototype.reset=function(){this.renderOrigin.set(this.origin),this.renderFront.set(this.front),this.pathCommands.forEach((function(t){t.reset()}))},i.prototype.transform=function(t,e,r){this.renderOrigin.transform(t,e,r),this.renderFront.transform(t,e,r),this.renderNormal.set(this.renderOrigin).subtract(this.renderFront),this.pathCommands.forEach((function(n){n.transform(t,e,r)})),this.children.forEach((function(n){n.transform(t,e,r)}))},i.prototype.updateSortValue=function(){var t=this.pathCommands.length,e=this.pathCommands[0].endRenderPoint,r=this.pathCommands[t-1].endRenderPoint;t>2&&e.isSame(r)&&(t-=1);for(var n=0,i=0;i<t;i++)n+=this.pathCommands[i].endRenderPoint.z;this.sortValue=n/t},i.prototype.render=function(t,e){var r=this.pathCommands.length;if(this.visible&&r&&(this.isFacingBack=this.renderNormal.z>0,this.backface||!this.isFacingBack)){if(!e)throw new Error("Zdog renderer required. Set to "+e);var n=1==r;e.isCanvas&&n?this.renderCanvasDot(t,e):this.renderPath(t,e)}};var a=t.TAU;i.prototype.renderCanvasDot=function(t){var e=this.getLineWidth();if(e){t.fillStyle=this.getRenderColor();var r=this.pathCommands[0].endRenderPoint;t.beginPath();var n=e/2;t.arc(r.x,r.y,n,0,a),t.fill()}},i.prototype.getLineWidth=function(){return this.stroke?1==this.stroke?1:this.stroke:0},i.prototype.getRenderColor=function(){return"string"==typeof this.backface&&this.isFacingBack?this.backface:this.color},i.prototype.renderPath=function(t,e){var r=this.getRenderElement(t,e),n=!(2==this.pathCommands.length&&"line"==this.pathCommands[1].method)&&this.closed,i=this.getRenderColor();e.renderPath(t,r,this.pathCommands,n),e.stroke(t,r,this.stroke,i,this.getLineWidth()),e.fill(t,r,this.fill,i),e.end(t,r)};return i.prototype.getRenderElement=function(t,e){if(e.isSvg)return this.svgElement||(this.svgElement=document.createElementNS("http://www.w3.org/2000/svg","path"),this.svgElement.setAttribute("stroke-linecap","round"),this.svgElement.setAttribute("stroke-linejoin","round")),this.svgElement},i}))},U14w:function(t,e,r){!function(e,n){if(t.exports)t.exports=n(r("SnCn"));else{var i=e.Zdog;i.Rect=n(i.Shape)}}(this,(function(t){var e=t.subclass({width:1,height:1});return e.prototype.setPath=function(){var t=this.width/2,e=this.height/2;this.path=[{x:-t,y:-e},{x:t,y:-e},{x:t,y:e},{x:-t,y:e}]},e}))},X9wK:function(t,e,r){r("ma9I"),r("QWBl"),r("yXV3"),r("+2oP"),r("ToJy"),r("pDQq"),r("uL8W"),r("eoL8"),r("tkto"),r("FZtP"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("RtnT"),r("6ZIi"),r("v+tz"));else{var i=e.Zdog;i.Anchor=n(i,i.Vector,i.CanvasRenderer,i.SvgRenderer)}}(this,(function(t,e,r,n){var i=t.TAU,o={x:1,y:1,z:1};function a(t){this.create(t||{})}return a.prototype.create=function(r){this.children=[],t.extend(this,this.constructor.defaults),this.setOptions(r),this.translate=new e(r.translate),this.rotate=new e(r.rotate),this.scale=new e(o).multiply(this.scale),this.origin=new e,this.renderOrigin=new e,this.addTo&&this.addTo.addChild(this)},a.defaults={},a.optionKeys=Object.keys(a.defaults).concat(["rotate","translate","scale","addTo"]),a.prototype.setOptions=function(t){var e=this.constructor.optionKeys;for(var r in t)-1!=e.indexOf(r)&&(this[r]=t[r])},a.prototype.addChild=function(t){-1==this.children.indexOf(t)&&(t.remove(),t.addTo=this,this.children.push(t))},a.prototype.removeChild=function(t){var e=this.children.indexOf(t);-1!=e&&this.children.splice(e,1)},a.prototype.remove=function(){this.addTo&&this.addTo.removeChild(this)},a.prototype.update=function(){this.reset(),this.children.forEach((function(t){t.update()})),this.transform(this.translate,this.rotate,this.scale)},a.prototype.reset=function(){this.renderOrigin.set(this.origin)},a.prototype.transform=function(t,e,r){this.renderOrigin.transform(t,e,r),this.children.forEach((function(n){n.transform(t,e,r)}))},a.prototype.updateGraph=function(){this.update(),this.updateFlatGraph(),this.flatGraph.forEach((function(t){t.updateSortValue()})),this.flatGraph.sort(a.shapeSorter)},a.shapeSorter=function(t,e){return t.sortValue-e.sortValue},Object.defineProperty(a.prototype,"flatGraph",{get:function(){return this._flatGraph||this.updateFlatGraph(),this._flatGraph},set:function(t){this._flatGraph=t}}),a.prototype.updateFlatGraph=function(){this.flatGraph=this.getFlatGraph()},a.prototype.getFlatGraph=function(){var t=[this];return this.addChildFlatGraph(t)},a.prototype.addChildFlatGraph=function(t){return this.children.forEach((function(e){var r=e.getFlatGraph();Array.prototype.push.apply(t,r)})),t},a.prototype.updateSortValue=function(){this.sortValue=this.renderOrigin.z},a.prototype.render=function(){},a.prototype.renderGraphCanvas=function(t){if(!t)throw new Error("ctx is "+t+". Canvas context required for render. Check .renderGraphCanvas( ctx ).");this.flatGraph.forEach((function(e){e.render(t,r)}))},a.prototype.renderGraphSvg=function(t){if(!t)throw new Error("svg is "+t+". SVG required for render. Check .renderGraphSvg( svg ).");this.flatGraph.forEach((function(e){e.render(t,n)}))},a.prototype.copy=function(e){var r={};return this.constructor.optionKeys.forEach((function(t){r[t]=this[t]}),this),t.extend(r,e),new(0,this.constructor)(r)},a.prototype.copyGraph=function(t){var e=this.copy(t);return this.children.forEach((function(t){t.copyGraph({addTo:e})})),e},a.prototype.normalizeRotate=function(){this.rotate.x=t.modulo(this.rotate.x,i),this.rotate.y=t.modulo(this.rotate.y,i),this.rotate.z=t.modulo(this.rotate.z,i)},a.subclass=function e(r){return function(n){function i(t){this.create(t||{})}return i.prototype=Object.create(r.prototype),i.prototype.constructor=i,i.defaults=t.extend({},r.defaults),t.extend(i.defaults,n),i.optionKeys=r.optionKeys.slice(0),Object.keys(i.defaults).forEach((function(t){1!=!i.optionKeys.indexOf(t)&&i.optionKeys.push(t)})),i.subclass=e(i),i}}(a),a}))},Y0B8:function(t,e,r){r("QWBl"),r("2B1R"),r("FZtP"),function(e,n){if(t.exports)t.exports=n(r("RtnT"));else{var i=e.Zdog;i.PathCommand=n(i.Vector)}}(this,(function(t){function e(e,i,o){this.method=e,this.points=i.map(r),this.renderPoints=i.map(n),this.previousPoint=o,this.endRenderPoint=this.renderPoints[this.renderPoints.length-1],"arc"==e&&(this.controlPoints=[new t,new t])}function r(e){return e instanceof t?e:new t(e)}function n(e){return new t(e)}e.prototype.reset=function(){var t=this.points;this.renderPoints.forEach((function(e,r){var n=t[r];e.set(n)}))},e.prototype.transform=function(t,e,r){this.renderPoints.forEach((function(n){n.transform(t,e,r)}))},e.prototype.render=function(t,e,r){return this[this.method](t,e,r)},e.prototype.move=function(t,e,r){return r.move(t,e,this.renderPoints[0])},e.prototype.line=function(t,e,r){return r.line(t,e,this.renderPoints[0])},e.prototype.bezier=function(t,e,r){var n=this.renderPoints[0],i=this.renderPoints[1],o=this.renderPoints[2];return r.bezier(t,e,n,i,o)};return e.prototype.arc=function(t,e,r){var n=this.previousPoint,i=this.renderPoints[0],o=this.renderPoints[1],a=this.controlPoints[0],s=this.controlPoints[1];return a.set(n).lerp(i,9/16),s.set(o).lerp(i,9/16),r.bezier(t,e,a,s,o)},e}))},hmEX:function(t,e,r){var n,i;n=this,i=function(){var t={};t.TAU=2*Math.PI,t.extend=function(t,e){for(var r in e)t[r]=e[r];return t},t.lerp=function(t,e,r){return(e-t)*r+t},t.modulo=function(t,e){return(t%e+e)%e};var e={2:function(t){return t*t},3:function(t){return t*t*t},4:function(t){return t*t*t*t},5:function(t){return t*t*t*t*t}};return t.easeInOut=function(t,r){if(1==r)return t;var n=(t=Math.max(0,Math.min(1,t)))<.5,i=n?t:1-t,o=(e[r]||e[2])(i/=.5);return o/=2,n?o:1-o},t},t.exports?t.exports=i():n.Zdog=i()},lP2j:function(t,e,r){!function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("SnCn"));else{var i=e.Zdog;i.Polygon=n(i,i.Shape)}}(this,(function(t,e){var r=e.subclass({sides:3,radius:.5}),n=t.TAU;return r.prototype.setPath=function(){this.path=[];for(var t=0;t<this.sides;t++){var e=t/this.sides*n-n/4,r=Math.cos(e)*this.radius,i=Math.sin(e)*this.radius;this.path.push({x:r,y:i})}},r}))},oqrJ:function(t,e,r){!function(e,n){if(t.exports)t.exports=n(r("SnCn"));else{var i=e.Zdog;i.Ellipse=n(i.Shape)}}(this,(function(t){var e=t.subclass({diameter:1,width:void 0,height:void 0,quarters:4,closed:!1});return e.prototype.setPath=function(){var t=(null!=this.width?this.width:this.diameter)/2,e=(null!=this.height?this.height:this.diameter)/2;this.path=[{x:0,y:-e},{arc:[{x:t,y:-e},{x:t,y:0}]}],this.quarters>1&&this.path.push({arc:[{x:t,y:e},{x:0,y:e}]}),this.quarters>2&&this.path.push({arc:[{x:-t,y:e},{x:-t,y:0}]}),this.quarters>3&&this.path.push({arc:[{x:-t,y:-e},{x:0,y:-e}]})},e}))},uPLu:function(t,e,r){r("yyme"),r("QWBl"),r("eoL8"),function(e,n){if(t.exports)t.exports=n(r("hmEX"),r("X9wK"),r("SnCn"),r("U14w"));else{var i=e.Zdog;i.Box=n(i,i.Anchor,i.Shape,i.Rect)}}(this,(function(t,e,r,n){var i=n.subclass();i.prototype.copyGraph=function(){};var o=t.TAU,a=["frontFace","rearFace","leftFace","rightFace","topFace","bottomFace"],s=t.extend({},r.defaults);delete s.path,a.forEach((function(t){s[t]=!0})),t.extend(s,{width:1,height:1,depth:1,fill:!0});var h=e.subclass(s);h.prototype.create=function(t){e.prototype.create.call(this,t),this.updatePath(),this.fill=this.fill},h.prototype.updatePath=function(){a.forEach((function(t){this[t]=this[t]}),this)},a.forEach((function(t){var e="_"+t;Object.defineProperty(h.prototype,t,{get:function(){return this[e]},set:function(r){this[e]=r,this.setFace(t,r)}})})),h.prototype.setFace=function(t,e){var r=t+"Rect",n=this[r];if(e){var o=this.getFaceOptions(t);o.color="string"==typeof e?e:this.color,n?n.setOptions(o):n=this[r]=new i(o),n.updatePath(),this.addChild(n)}else this.removeChild(n)},h.prototype.getFaceOptions=function(t){return{frontFace:{width:this.width,height:this.height,translate:{z:this.depth/2}},rearFace:{width:this.width,height:this.height,translate:{z:-this.depth/2},rotate:{y:o/2}},leftFace:{width:this.depth,height:this.height,translate:{x:-this.width/2},rotate:{y:-o/4}},rightFace:{width:this.depth,height:this.height,translate:{x:this.width/2},rotate:{y:o/4}},topFace:{width:this.width,height:this.depth,translate:{y:-this.height/2},rotate:{x:-o/4}},bottomFace:{width:this.width,height:this.depth,translate:{y:this.height/2},rotate:{x:o/4}}}[t]};return["color","stroke","fill","backface","front","visible"].forEach((function(t){var e="_"+t;Object.defineProperty(h.prototype,t,{get:function(){return this[e]},set:function(r){this[e]=r,a.forEach((function(e){var n=this[e+"Rect"],i="string"==typeof this[e];n&&!("color"==t&&i)&&(n[t]=r)}),this)}})})),h}))},"v+tz":function(t,e,r){var n,i;r("QWBl"),r("FZtP"),n=this,i=function(){var t={isSvg:!0},e=t.round=function(t){return Math.round(1e3*t)/1e3};function r(t){return e(t.x)+","+e(t.y)+" "}return t.begin=function(){},t.move=function(t,e,n){return"M"+r(n)},t.line=function(t,e,n){return"L"+r(n)},t.bezier=function(t,e,n,i,o){return"C"+r(n)+r(i)+r(o)},t.closePath=function(){return"Z"},t.setPath=function(t,e,r){e.setAttribute("d",r)},t.renderPath=function(e,r,n,i){var o="";n.forEach((function(n){o+=n.render(e,r,t)})),i&&(o+=this.closePath(e,r)),this.setPath(e,r,o)},t.stroke=function(t,e,r,n,i){r&&(e.setAttribute("stroke",n),e.setAttribute("stroke-width",i))},t.fill=function(t,e,r,n){var i=r?n:"none";e.setAttribute("fill",i)},t.end=function(t,e){t.appendChild(e)},t},t.exports?t.exports=i():n.Zdog.SvgRenderer=i()}}]);
//# sourceMappingURL=component---src-pages-analyze-js-55fb03378688a31cdc58.js.map