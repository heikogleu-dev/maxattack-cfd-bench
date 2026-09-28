#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



// ################################################## utility functions ##################################################

)+R(float sq(const float x) {
	return x*x;
}
)+R(float cb(const float x) {
	return x*x*x;
}
)+R(float angle(const float3 v1, const float3 v2) {
	return acos(dot(v1, v2)/(length(v1)*length(v2)));
}
)+R(float fast_rsqrt(const float x) { // slightly faster approximation
	return as_float(0x5F37642F-(as_int(x)>>1));
}
)+R(float fast_asin(const float x) { // slightly faster approximation
	return x*fma(0.5702f, sq(sq(sq(x))), 1.0f); // 0.5707964f = (pi-2)/2
}
)+R(float fast_acos(const float x) { // slightly faster approximation
	return fma(fma(-0.5702f, sq(sq(sq(x))), -1.0f), x, 1.5712963f); // 0.5707964f = (pi-2)/2
}
)+R(void swap(float* x, float* y) {
	const float t = *x;
	*x = *y;
	*y = t;
}
)+R(void lu_solve(float* M, float* x, float* b, const int N, const int Nsol) { // solves system of N linear equations M*x=b within dimensionality Nsol<=N
	for(int i=0; i<Nsol; i++) { // decompose M in M=L*U
		for(int j=i+1; j<Nsol; j++) {
			M[N*j+i] /= M[N*i+i];
			for(int k=i+1; k<Nsol; k++) M[N*j+k] -= M[N*j+i]*M[N*i+k];
		}
	}
	for(int i=0; i<Nsol; i++) { // find solution of L*y=b
		x[i] = b[i];
		for(int k=0; k<i; k++) x[i] -= M[N*i+k]*x[k];
	}
	for(int i=Nsol-1; i>=0; i--) { // find solution of U*x=y
		for(int k=i+1; k<Nsol; k++) x[i] -= M[N*i+k]*x[k];
		x[i] /= M[N*i+i];
	}
}
)+R(float trilinear(const float3 p, const float* v) { // trilinear interpolation, p: position in unit cube, v: corner values in unit cube
	const float x1=p.x, y1=p.y, z1=p.z, x0=1.0f-x1, y0=1.0f-y1, z0=1.0f-z1; // calculate interpolation factors
	return (x0*y0*z0)*v[0]+(x1*y0*z0)*v[1]+(x1*y0*z1)*v[2]+(x0*y0*z1)*v[3]+(x0*y1*z0)*v[4]+(x1*y1*z0)*v[5]+(x1*y1*z1)*v[6]+(x0*y1*z1)*v[7]; // perform trilinear interpolation
}
)+R(float3 trilinear3(const float3 p, const float3* v) { // trilinear interpolation, p: position in unit cube, v: corner vectors in unit cube
	const float x1=p.x, y1=p.y, z1=p.z, x0=1.0f-x1, y0=1.0f-y1, z0=1.0f-z1; // calculate interpolation factors
	return (x0*y0*z0)*v[0]+(x1*y0*z0)*v[1]+(x1*y0*z1)*v[2]+(x0*y0*z1)*v[3]+(x0*y1*z0)*v[4]+(x1*y1*z0)*v[5]+(x1*y1*z1)*v[6]+(x0*y1*z1)*v[7]; // perform trilinear interpolation
}



// ################################################## Line3D code ##################################################

// Line3D OpenCL C version (c) Dr. Moritz Lehmann
// draw_point(...)    : draw 3D pixel
// draw_circle(...)   : draw 3D circle
// draw_line(...)     : draw 3D line
// draw_triangle(...) : draw 3D triangle
// graphics_clear()   : kernel to reset bitmap and zbuffer

)+"#ifdef GRAPHICS"+R(
)+R(int color_from_floats(const float red, const float green, const float blue) {
	return clamp((int)fma(255.0f, red, 0.5f), 0, 255)<<16|clamp((int)fma(255.0f, green, 0.5f), 0, 255)<<8|clamp((int)fma(255.0f, blue, 0.5f), 0, 255);
}
)+R(int color_mul(const int c, const float x) { // c*x
	const int r = min((int)fma((float)((c>>16)&255), x, 0.5f), 255);
	const int g = min((int)fma((float)((c>> 8)&255), x, 0.5f), 255);
	const int b = min((int)fma((float)( c     &255), x, 0.5f), 255);
	return r<<16|g<<8|b; // values are already clamped
}
)+R(int color_average(const int c1, const int c2) { // (c1+c2)/s
	const uchar4 cc1=as_uchar4(c1), cc2=as_uchar4(c2);
	return as_int((uchar4)((uchar)((cc1.x+cc2.x)/2u), (uchar)((cc1.y+cc2.y)/2u), (uchar)((cc1.z+cc2.z)/2u), (uchar)0u));
}
)+R(int color_mix(const int c1, const int c2, const float w) { // w*c1+(1-w)*c2
	const uchar4 cc1=as_uchar4(c1), cc2=as_uchar4(c2);
	const float3 fc1=(float3)((float)cc1.x, (float)cc1.y, (float)cc1.z), fc2=(float3)((float)cc2.x, (float)cc2.y, (float)cc2.z);
	const float3 fcm = fma(w, fc1, fma(1.0f-w, fc2, (float3)(0.5f, 0.5f, 0.5f)));
	return as_int((uchar4)((uchar)fcm.x, (uchar)fcm.y, (uchar)fcm.z, (uchar)0u));
}
)+R(int color_mix_3(const int c0, const int c1, const int c2, const float w0, const float w1, const float w2) { // w1*c1+w2*c2+w3*c3, w0+w1+w2 = 1
	const uchar4 cc0=as_uchar4(c0), cc1=as_uchar4(c1), cc2=as_uchar4(c2);
	const float3 fc0=(float3)((float)cc0.x, (float)cc0.y, (float)cc0.z),  fc1=(float3)((float)cc1.x, (float)cc1.y, (float)cc1.z), fc2=(float3)((float)cc2.x, (float)cc2.y, (float)cc2.z);
	const float3 fcm = fma(w0, fc0, fma(w1, fc1, fma(w2, fc2, (float3)(0.5f, 0.5f, 0.5f))));
	return as_int((uchar4)((uchar)fcm.x, (uchar)fcm.y, (uchar)fcm.z, (uchar)0u));
}
)+R(int hsv_to_rgb(const float h, const float s, const float v) {
	const float c = v*s;
	const float x = c*(1.0f-fabs(fmod(h/60.0f, 2.0f)-1.0f));
	const float m = v-c;
	float r=0.0f, g=0.0f, b=0.0f;
	if(0.0f<=h&&h<60.0f) { r = c; g = x; }
	else if(h<120.0f) { r = x; g = c; }
	else if(h<180.0f) { g = c; b = x; }
	else if(h<240.0f) { g = x; b = c; }
	else if(h<300.0f) { r = x; b = c; }
	else if(h<360.0f) { r = c; b = x; }
	return color_from_floats(r+m, g+m, b+m);
}
)+R(int colorscale_rainbow(float x) { // coloring scheme (float [0, 1] -> int color)
	x = clamp(6.0f*(1.0f-x), 0.0f, 6.0f);
	float r=0.0f, g=0.0f, b=0.0f; // black
	if(x<1.2f) { // red - yellow
		r = 1.0f;
		g = x*0.83333333f;
	} else if(x>=1.2f&&x<2.0f) { // yellow - green
		r = 2.5f-x*1.25f;
		g = 1.0f;
	} else if(x>=2.0f&&x<3.0f) { // green - cyan
		g = 1.0f;
		b = x-2.0f;
	} else if(x>=3.0f&&x<4.0f) { // cyan - blue
		g = 4.0f-x;
		b = 1.0f;
	} else if(x>=4.0f&&x<5.0f) { // blue - violet
		r = x*0.4f-1.6f;
		b = 3.0f-x*0.5f;
	} else { // violet - black
		r = 2.4f-x*0.4f;
		b = 3.0f-x*0.5f;
	}
	return color_from_floats(r, g, b);
}
)+R(int colorscale_iron(float x) { // coloring scheme (float [0, 1] -> int color)
	x = clamp(4.0f*(1.0f-x), 0.0f, 4.0f);
	float r=1.0f, g=0.0f, b=0.0f;
	if(x<0.66666667f) { // white - yellow
		g = 1.0f;
		b = 1.0f-x*1.5f;
	} else if(x<2.0f) { // yellow - red
		g = 1.5f-x*0.75f;
	} else if(x<3.0f) { // red - violet
		r = 2.0f-x*0.5f;
		b = x-2.0f;
	} else { // violet - black
		r = 2.0f-x*0.5f;
		b = 4.0f-x;
	}
	return color_from_floats(r, g, b);
}
)+R(int colorscale_twocolor(float x) { // coloring scheme (float [0, 1] -> int color)
	return x>0.5f ? color_mix(0xFFAA00, def_background_color, clamp(2.0f*x-1.0f, 0.0f, 1.0f)) : color_mix(def_background_color, 0x0080FF, clamp(2.0f*x, 0.0f, 1.0f)); // red - gray - blue
}
)+R(int shading(const int c, const float3 p, const float3 normal, const float* camera_cache) { // calculate flat shading color of triangle
)+"#ifndef GRAPHICS_TRANSPARENCY"+R(
	const float zoom = camera_cache[ 0]; // fetch camera parameters (rotation matrix, camera position, etc.)
	const float  dis = camera_cache[ 1];
	const float3 pos = (float3)(camera_cache[ 2], camera_cache[ 3], camera_cache[ 4])-(float3)(def_domain_offset_x, def_domain_offset_y, def_domain_offset_z);
	const float3 Rz  = (float3)(camera_cache[11], camera_cache[12], camera_cache[13]);
	const float3 d = p-Rz*(dis/zoom)-pos; // distance vector between p and camera position
	const float nl2 = fma(normal.x, normal.x, fma(normal.y, normal.y, sq(normal.z))); // only one rsqrt instead of two
	const float dl2 = fma(d.x, d.x, fma(d.y, d.y, sq(d.z)));
	return color_mul(c, max(1.25f*fabs(dot(normal, d))*rsqrt(nl2*dl2), 0.3f));
)+"#else"+R( // GRAPHICS_TRANSPARENCY
	return c; // disable flat shading, just return input color
)+"#endif"+R( // GRAPHICS_TRANSPARENCY
}
)+R(bool is_off_screen(const int x, const int y, const int stereo) {
	switch(stereo) {
		default: return x<                 0||x>=def_screen_width  ||y<0||y>=def_screen_height; // entire screen
		case -1: return x<                 0||x>=def_screen_width/2||y<0||y>=def_screen_height; // left half
		case +1: return x<def_screen_width/2||x>=def_screen_width  ||y<0||y>=def_screen_height; // right half
	}
}
)+R(void draw(const int x, const int y, const float z, const int color, volatile global int* bitmap, volatile global int* zbuffer, const int stereo) {
	const int index=x+y*def_screen_width, iz=(int)(z*1E3f); // use fixed-point int z-buffer and atomic_max to minimize noise in image, maximum render distance is 2.147E6f
	if(!is_off_screen(x, y, stereo)) { // only draw if point is on screen
)+"#ifndef GRAPHICS_TRANSPARENCY"+R(
		if(iz>atomic_max(&zbuffer[index], iz)) atomic_xchg(&bitmap[index], color); // only draw if point is first in zbuffer
)+"#else"+R( // GRAPHICS_TRANSPARENCY
		const float transparency = GRAPHICS_TRANSPARENCY; // transparent rendering (not quite order-independent transparency, but elegant solution for order-reversible transparency which is good enough here)
		const uchar4 cc4=as_uchar4(color), cb4=as_uchar4(def_background_color);
		const float3 fc = (float3)((float)cc4.x, (float)cc4.y, (float)cc4.z); // new pixel color that is behind topmost drawn pixel color
		const float3 fb = (float3)((float)cb4.x, (float)cb4.y, (float)cb4.z); // background color
		const uchar4 cp4 = as_uchar4(bitmap[index]);
		const float3 fp = (float3)((float)cp4.x, (float)cp4.y, (float)cp4.z); // current pixel color
		const int draw_count = (int)cp4.w; // use alpha color value to store how often the pixel has been over-drawn already
		const float3 fn = fp+(1.0f-transparency)*(iz>atomic_max(&zbuffer[index], iz) ? fc-fp : pown(transparency, draw_count)*(fc-fb)); // black magic: either over-draw colors back-to-front, or add back colors as correction terms
		atomic_xchg(&bitmap[index], as_int((uchar4)((uchar)clamp(fn.x+0.5f, 0.0f, 255.0f), (uchar)clamp(fn.y+0.5f, 0.0f, 255.0f), (uchar)clamp(fn.z+0.5f, 0.0f, 255.0f), (uchar)min(draw_count+1, 255))));
)+"#endif"+R( // GRAPHICS_TRANSPARENCY
	}
}
)+R(bool convert(int* rx, int* ry, float* rz, const float3 p, const float* camera_cache, const int stereo) { // 3D -> 2D
	const float zoom = camera_cache[0]; // fetch camera parameters (rotation matrix, camera position, etc.)
	const float  dis = camera_cache[1];
	const float3 pos = (float3)(camera_cache[ 2], camera_cache[ 3], camera_cache[ 4])-(float3)(def_domain_offset_x, def_domain_offset_y, def_domain_offset_z);
	const float3 Rx  = (float3)(camera_cache[ 5], camera_cache[ 6], camera_cache[ 7]);
	const float3 Ry  = (float3)(camera_cache[ 8], camera_cache[ 9], camera_cache[10]);
	const float3 Rz  = (float3)(camera_cache[11], camera_cache[12], camera_cache[13]);
	const float eye_distance = vload_half(28, (half*)camera_cache);
	float3 t, r;
	t = p-pos-((float)stereo*eye_distance/zoom)*(float3)(Rx.x, Rx.y, 0.0f); // transformation
	r.z = dot(Rz, t); // z-position for z-buffer
	const float rs = zoom*dis/(dis-r.z*zoom); // perspective (reciprocal is more efficient)
	if(rs<=0.0f) return false; // point is behins camera
	const float tv = ((as_int(camera_cache[14])>>30)&0x1)&&stereo!=0 ? 0.5f : 1.0f;
	r.x = (dot(Rx, t)*rs+(float)stereo*eye_distance)*tv+(0.5f+(float)stereo*0.25f)*(float)def_screen_width; // x position on screen
	r.y =  dot(Ry, t)*rs+0.5f*(float)def_screen_height; // y position on screen
	*rx = (int)(r.x+0.5f);
	*ry = (int)(r.y+0.5f);
	*rz = r.z;
	return true;
}
)+R(void convert_circle(float3 p, const float r, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer, const int stereo) { // 3D -> 2D
	int rx, ry; float rz;
	if(convert(&rx, &ry, &rz, p, camera_cache, stereo)) {
		const float zoom = camera_cache[0];
		const float dis  = camera_cache[1];
		const float rs = zoom*dis/(dis-rz*zoom);
		const int radius = (int)(rs*r+0.5f);
		switch(stereo) {
			default: if(rx<                       -radius||rx>=(int)def_screen_width  +radius || ry<-radius||ry>=(int)def_screen_height+radius) return; break; // cancel drawing if circle is off screen
			case -1: if(rx<                       -radius||rx>=(int)def_screen_width/2+radius || ry<-radius||ry>=(int)def_screen_height+radius) return; break;
			case +1: if(rx<(int)def_screen_width/2-radius||rx>=(int)def_screen_width  +radius || ry<-radius||ry>=(int)def_screen_height+radius) return; break;
		}
		int d=-radius, x=radius, y=0; // Bresenham algorithm for circle
		while(x>=y) {
			draw(rx+x, ry+y, rz, color, bitmap, zbuffer, stereo);
			draw(rx-x, ry+y, rz, color, bitmap, zbuffer, stereo);
			draw(rx+x, ry-y, rz, color, bitmap, zbuffer, stereo);
			draw(rx-x, ry-y, rz, color, bitmap, zbuffer, stereo);
			draw(rx+y, ry+x, rz, color, bitmap, zbuffer, stereo);
			draw(rx-y, ry+x, rz, color, bitmap, zbuffer, stereo);
			draw(rx+y, ry-x, rz, color, bitmap, zbuffer, stereo);
			draw(rx-y, ry-x, rz, color, bitmap, zbuffer, stereo);
			d += 2*y+1;
			y++;
			if(d>0) d-=2*(--x);
		}
	}
}
)+R(void convert_line(const float3 p0, const float3 p1, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer, const int stereo) { // 3D -> 2D
	int r0x, r0y, r1x, r1y; float r0z, r1z;
	if(convert(&r0x, &r0y, &r0z, p0, camera_cache, stereo) && convert(&r1x, &r1y, &r1z, p1, camera_cache, stereo)
		&& !(is_off_screen(r0x, r0y, stereo) && is_off_screen(r1x, r1y, stereo))) { // cancel drawing if both points are off screen
		int x=r0x, y=r0y; // Bresenham algorithm
		const float z = 0.5f*(r0z+r1z); // approximate line z position for each pixel to be equal
		const int dx= abs(r1x-r0x), sx=2*(r0x<r1x)-1;
		const int dy=-abs(r1y-r0y), sy=2*(r0y<r1y)-1;
		int err = dx+dy;
		while(x!=r1x||y!=r1y) {
			draw(x, y, z, color, bitmap, zbuffer, stereo);
			const int e2 = 2*err;
			if(e2>dy) { err+=dy; x+=sx; }
			if(e2<dx) { err+=dx; y+=sy; }
		}
	}
}
)+R(void convert_triangle(float3 p0, float3 p1, float3 p2, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer, const int stereo) { // 3D -> 2D
	int r0x, r0y, r1x, r1y, r2x, r2y; float r0z, r1z, r2z;
	if(convert(&r0x, &r0y, &r0z, p0, camera_cache, stereo) && convert(&r1x, &r1y, &r1z, p1, camera_cache, stereo) && convert(&r2x, &r2y, &r2z, p2, camera_cache, stereo)
		&& !(is_off_screen(r0x, r0y, stereo) && is_off_screen(r1x, r1y, stereo) && is_off_screen(r2x, r2y, stereo))) { // cancel drawing if all points are off screen
		if(r0x*(r1y-r2y)+r1x*(r2y-r0y)+r2x*(r0y-r1y)>40000 || (r0y==r1y&&r0y==r2y)) return; // return for large triangle area or degenerate triangles
		//if(r1x*r0y+r2x*r1y+r0x*r2y>=r0x*r1y+r1x*r2y+r2x*r0y) return; // clockwise backface culling
		if(r0y>r1y) { const int xt = r0x; const int yt = r0y; r0x = r1x; r0y = r1y; r1x = xt; r1y = yt; } // sort vertices ascending by y
		if(r0y>r2y) { const int xt = r0x; const int yt = r0y; r0x = r2x; r0y = r2y; r2x = xt; r2y = yt; }
		if(r1y>r2y) { const int xt = r1x; const int yt = r1y; r1x = r2x; r1y = r2y; r2x = xt; r2y = yt; }
		const float z = (r0z+r1z+r2z)/3.0f; // approximate triangle z position for each pixel to be equal
		const int r2xr0x=r2x-r0x, r2yr0y=r2y-r0y, r1xr0x=r1x-r0x, r1yr0y=r1y-r0y, r2xr1x=r2x-r1x, r2yr1y=r2y-r1y;
		for(int y=r0y; y<r1y; y++) { // Bresenham algorithm (lower triangle half)
			const int xA = r0x+r2xr0x*(y-r0y)/r2yr0y;
			const int xB = r0x+r1xr0x*(y-r0y)/r1yr0y;
			for(int x=min(xA, xB); x<max(xA, xB); x++) {
				draw(x, y, z, color, bitmap, zbuffer, stereo);
			}
		}
		for(int y=r1y; y<r2y; y++) { // Bresenham algorithm (upper triangle half)
			const int xA = r0x+r2xr0x*(y-r0y)/r2yr0y;
			const int xB = r1x+r2xr1x*(y-r1y)/r2yr1y;
			for(int x=min(xA, xB); x<max(xA, xB); x++) {
				draw(x, y, z, color, bitmap, zbuffer, stereo);
			}
		}
	}
}
)+R(void convert_triangle_interpolated(float3 p0, float3 p1, float3 p2, int c0, int c1, int c2, const float* camera_cache, global int* bitmap, global int* zbuffer, const int stereo) { // 3D -> 2D
	int r0x, r0y, r1x, r1y, r2x, r2y; float r0z, r1z, r2z;
	if(convert(&r0x, &r0y, &r0z, p0, camera_cache, stereo) && convert(&r1x, &r1y, &r1z, p1, camera_cache, stereo) && convert(&r2x, &r2y, &r2z, p2, camera_cache, stereo)
		&& !(is_off_screen(r0x, r0y, stereo) && is_off_screen(r1x, r1y, stereo) && is_off_screen(r2x, r2y, stereo))) { // cancel drawing if all points are off screen
		if(r0x*(r1y-r2y)+r1x*(r2y-r0y)+r2x*(r0y-r1y)>40000 || (r0y==r1y&&r0y==r2y)) return; // return for large triangle area or degenerate triangles
		//if(r1x*r0y+r2x*r1y+r0x*r2y>=r0x*r1y+r1x*r2y+r2x*r0y) return; // clockwise backface culling
		if(r0y>r1y) { const int xt = r0x; const int yt = r0y; r0x = r1x; r0y = r1y; r1x = xt; r1y = yt; const int ct = c0; c0 = c1; c1 = ct; } // sort vertices ascending by y
		if(r0y>r2y) { const int xt = r0x; const int yt = r0y; r0x = r2x; r0y = r2y; r2x = xt; r2y = yt; const int ct = c0; c0 = c2; c2 = ct; }
		if(r1y>r2y) { const int xt = r1x; const int yt = r1y; r1x = r2x; r1y = r2y; r2x = xt; r2y = yt; const int ct = c1; c1 = c2; c2 = ct; }
		const float z = (r0z+r1z+r2z)/3.0f; // approximate triangle z position for each pixel to be equal
		const int r0xr2x=r0x-r2x, r2xr0x=-r0xr2x;
		const int r0yr2y=r0y-r2y, r2yr0y=-r0yr2y;
		const int r1yr2y=r1y-r2y, r2yr1y=-r1yr2y;
		const int r1xr0x=r1x-r0x, r1yr0y=r1y-r0y, r2xr1x=r2x-r1x;
		const int cw0=r1yr2y*r2x+r2xr1x*r2y, cw1=r2yr0y*r2x+r0xr2x*r2y;
		const float d = 1.0f/(float)(r1yr2y*r0xr2x+r2xr1x*r0yr2y);
		const uchar4 cc0=as_uchar4(c0), cc1=as_uchar4(c1), cc2=as_uchar4(c2);
		const float3 fc0=(float3)((float)cc0.x, (float)cc0.y, (float)cc0.z),  fc1=(float3)((float)cc1.x, (float)cc1.y, (float)cc1.z), fc2=(float3)((float)cc2.x, (float)cc2.y, (float)cc2.z);
		for(int y=r0y; y<r1y; y++) { // Bresenham algorithm (lower triangle half)
			const int xA = r0x+r2xr0x*(y-r0y)/r2yr0y;
			const int xB = r0x+r1xr0x*(y-r0y)/r1yr0y;
			for(int x=min(xA, xB); x<max(xA, xB); x++) {
				const float w0 = (float)(r1yr2y*x+r2xr1x*y-cw0)*d; // barycentric coordinates
				const float w1 = (float)(r2yr0y*x+r0xr2x*y-cw1)*d;
				const float w2 = 1.0f-w0-w1;
				const float3 fcm = fma(w0, fc0, fma(w1, fc1, fma(w2, fc2, (float3)(0.5f, 0.5f, 0.5f)))); // interpolate color
				const int color = as_int((uchar4)((uchar)fcm.x, (uchar)fcm.y, (uchar)fcm.z, (uchar)0u));
				draw(x, y, z, color, bitmap, zbuffer, stereo);
			}
		}
		for(int y=r1y; y<r2y; y++) { // Bresenham algorithm (upper triangle half)
			const int xA = r0x+r2xr0x*(y-r0y)/r2yr0y;
			const int xB = r1x+r2xr1x*(y-r1y)/r2yr1y;
			for(int x=min(xA, xB); x<max(xA, xB); x++) {
				const float w0 = (float)(r1yr2y*x+r2xr1x*y-cw0)*d; // barycentric coordinates
				const float w1 = (float)(r2yr0y*x+r0xr2x*y-cw1)*d;
				const float w2 = 1.0f-w0-w1;
				const float3 fcm = fma(w0, fc0, fma(w1, fc1, fma(w2, fc2, (float3)(0.5f, 0.5f, 0.5f)))); // interpolate color
				const int color = as_int((uchar4)((uchar)fcm.x, (uchar)fcm.y, (uchar)fcm.z, (uchar)0u));
				draw(x, y, z, color, bitmap, zbuffer, stereo);
			}
		}
	}
}
)+R(void draw_point(const float3 p, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer) { // 3D -> 2D
	const int vr = (as_int(camera_cache[14])>>31)&0x1;
	int rx, ry; float rz;
	if(vr&&convert(&rx, &ry, &rz, p, camera_cache, -1)) draw(rx, ry, rz, color, bitmap, zbuffer, -1); // left eye
	if(/**/convert(&rx, &ry, &rz, p, camera_cache, vr)) draw(rx, ry, rz, color, bitmap, zbuffer, vr); // right eye
}
)+R(void draw_circle(const float3 p, const float r, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer) { // 3D -> 2D
	const int vr = (as_int(camera_cache[14])>>31)&0x1;
	if(vr) convert_circle(p, r, color, camera_cache, bitmap, zbuffer, -1); // left eye
	/****/ convert_circle(p, r, color, camera_cache, bitmap, zbuffer, vr); // right eye
}
)+R(void draw_line(const float3 p0, const float3 p1, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer) { // 3D -> 2D
	const int vr = (as_int(camera_cache[14])>>31)&0x1;
	if(vr) convert_line(p0, p1, color, camera_cache, bitmap, zbuffer, -1); // left eye
	/****/ convert_line(p0, p1, color, camera_cache, bitmap, zbuffer, vr); // right eye
}
)+R(void draw_triangle(const float3 p0, const float3 p1, const float3 p2, const int color, const float* camera_cache, global int* bitmap, global int* zbuffer) { // 3D -> 2D
	const int vr = (as_int(camera_cache[14])>>31)&0x1;
	if(vr) convert_triangle(p0, p1, p2, color, camera_cache, bitmap, zbuffer, -1); // left eye
	/****/ convert_triangle(p0, p1, p2, color, camera_cache, bitmap, zbuffer, vr); // right eye
}
)+R(__attribute__((always_inline)) void draw_triangle_interpolated(const float3 p0, const float3 p1, const float3 p2, const int c0, const int c1, const int c2, const float* camera_cache, global int* bitmap, global int* zbuffer) { // 3D -> 2D
	const int vr = (as_int(camera_cache[14])>>31)&0x1;
	if(vr) convert_triangle_interpolated(p0, p1, p2, c0, c1, c2, camera_cache, bitmap, zbuffer, -1); // left eye
	/****/ convert_triangle_interpolated(p0, p1, p2, c0, c1, c2, camera_cache, bitmap, zbuffer, vr); // right eye
}
)+R(kernel void graphics_clear(global int* bitmap, global int* zbuffer) {
	const uint n = get_global_id(0);
	bitmap[n] = def_background_color; // black background = 0x000000, use 0xFFFFFF for white background
	zbuffer[n] = -2147483648;
}
)+R(constant uchar triangle_table_data[1920] = { // source: Paul Bourke, http://paulbourke.net/geometry/polygonise/, termination value 15, bit packed
	255,255,255,255,255,255,255, 15, 56,255,255,255,255,255,255, 16,249,255,255,255,255,255, 31, 56,137,241,255,255,255,255, 33,250,255,255,255,255,255, 15, 56, 33,250,255,255,255,255, 41, 10,146,
	255,255,255,255, 47, 56,162,168,137,255,255,255,179,242,255,255,255,255,255, 15, 43,184,240,255,255,255,255,145, 32,179,255,255,255,255, 31, 43,145,155,184,255,255,255,163,177, 58,255,255,255,
	255, 15, 26,128,138,171,255,255,255,147, 48,155,171,249,255,255,159,168,138,251,255,255,255,255,116,248,255,255,255,255,255, 79,  3, 55,244,255,255,255,255, 16,137,116,255,255,255,255, 79,145,
	116,113, 19,255,255,255, 33,138,116,255,255,255,255, 63,116,  3, 20,162,255,255,255, 41,154, 32, 72,247,255,255, 47,154,146, 39, 55,151,244,255, 72, 55, 43,255,255,255,255,191,116, 43, 36, 64,
	255,255,255,  9,129,116, 50,251,255,255, 79,183, 73,155, 43, 41,241,255,163, 49,171,135,244,255,255, 31,171, 65, 27, 64,183,244,255,116,152,176,185,186, 48,255, 79,183,180,153,171,255,255,255,
	 89,244,255,255,255,255,255,159, 69,128,243,255,255,255,255, 80, 20,  5,255,255,255,255,143, 69, 56, 53, 81,255,255,255, 33,154, 69,255,255,255,255, 63,128, 33, 74, 89,255,255,255, 37, 90, 36,
	  4,242,255,255, 47, 90, 35, 53, 69, 67,248,255, 89, 36,179,255,255,255,255, 15, 43,128, 75, 89,255,255,255, 80,  4, 81, 50,251,255,255, 47, 81, 82, 40,184,132,245,255, 58,171, 49, 89,244,255,
	255, 79, 89,128,129, 26,184,250,255, 69, 80,176,181,186, 48,255, 95,132,133,170,184,255,255,255,121, 88,151,255,255,255,255,159,  3, 89, 83, 55,255,255,255,112,  8,113, 81,247,255,255, 31, 53,
	 83,247,255,255,255,255,121,152,117, 26,242,255,255,175, 33, 89, 80,  3,117,243,255,  8,130, 82, 88,167, 37,255, 47, 90, 82, 51,117,255,255,255,151,117,152,179,242,255,255,159,117,121,146,  2,
	114,251,255, 50, 11,129,113, 24,117,255,191, 18, 27,119, 81,255,255,255, 89,136,117, 26,163,179,255, 95,  7,  5,121, 11,  1,186, 10,171,176, 48, 90,128,112,117,176, 90,183,245,255,255,255,255,
	106,245,255,255,255,255,255, 15, 56,165,246,255,255,255,255,  9, 81,106,255,255,255,255, 31, 56,145, 88,106,255,255,255, 97, 37, 22,255,255,255,255, 31, 86, 33, 54,128,255,255,255,105,149, 96,
	 32,246,255,255, 95,137,133, 82, 98, 35,248,255, 50,171, 86,255,255,255,255,191,128, 43,160, 86,255,255,255, 16, 41,179,165,246,255,255, 95,106,145,146, 43,137,251,255, 54,107, 53, 21,243,255,
	255, 15,184,176,  5, 21,181,246,255,179,  6, 99, 96,  5,149,255,111,149,150,187,137,255,255,255,165, 70,135,255,255,255,255, 79,  3,116, 99,165,255,255,255,145, 80,106, 72,247,255,255,175, 86,
	145, 23, 55,151,244,255, 22, 98, 21,116,248,255,255, 31, 82, 37, 54, 64, 67,247,255, 72,151, 80, 96,  5, 98,255,127,147,151, 52,146,149, 38,150,179,114, 72,106,245,255,255, 95,106,116, 66,  2,
	114,251,255, 16, 73,135, 50, 91,106,255,159, 18,185,146,180,183, 84,106, 72, 55, 91, 83, 81,107,255, 95,177,181, 22,176,183,  4,180, 80,  9, 86, 48,182, 54, 72,103,149,150, 75,151,183,249,255,
	 74,105,164,255,255,255,255, 79,106,148, 10, 56,255,255,255, 10,161,  6, 70,240,255,255,143, 19, 24,134, 70, 22,250,255, 65, 25, 66, 98,244,255,255, 63,128, 33, 41,148, 98,244,255, 32, 68, 98,
	255,255,255,255,143, 35, 40, 68, 98,255,255,255, 74,169, 70, 43,243,255,255, 15, 40,130, 75,169,164,246,255,179,  2, 97, 96,100,161,255,111, 20, 22, 74, 24, 18,139, 27,105,148, 99, 25,179, 54,
	255,143, 27, 24,176, 22, 25,100, 20,179, 54,  6, 96,244,255,255,111,132,107,248,255,255,255,255,167,118,168,152,250,255,255, 15, 55,160,  7,169,118,250,255,106, 23,122,113, 24,  8,255,175,118,
	122, 17, 55,255,255,255, 33, 22,134,129,137,118,255, 47,150,146, 97,151,144,115,147,135,112, 96,  6,242,255,255,127, 35,118,242,255,255,255,255, 50,171,134,138,137,118,255, 47,112,114, 11,121,
	118,154,122,129, 16,135,161,103,167, 50,187, 18, 27,167, 22,118,241,255,152,134,118, 25,182, 54, 49,  6, 25,107,247,255,255,255,255,135,112, 96,179,176,  6,255,127,107,255,255,255,255,255,255,
	103,251,255,255,255,255,255, 63,128,123,246,255,255,255,255, 16,185,103,255,255,255,255,143,145, 56,177,103,255,255,255, 26, 98,123,255,255,255,255, 31,162,  3,104,123,255,255,255,146, 32,154,
	182,247,255,255,111,123,162,163, 56,154,248,255, 39, 99,114,255,255,255,255,127,128,103, 96,  2,255,255,255,114, 38,115, 16,249,255,255, 31, 38,129, 22,137,120,246,255,122,166,113, 49,247,255,
	255,175,103,113, 26,120,  1,248,255, 48,  7,167,160,105,122,255,127,166,167,136,154,255,255,255,134,180,104,255,255,255,255, 63,182,  3,  6,100,255,255,255,104,139,100,  9,241,255,255,159,100,
	105,147, 19, 59,246,255,134,100,139,162,241,255,255, 31,162,  3, 11,182, 64,246,255,180, 72,182, 32, 41,154,255,175, 57, 58,146, 52, 59, 70, 54, 40,131, 36,100,242,255,255, 15, 36,100,242,255,
	255,255,255,145, 32, 67, 66, 70,131,255, 31, 73, 65, 34,100,255,255,255, 24,131, 22, 72,102, 26,255,175,  1, 10,102, 64,255,255,255,100, 67,131,166,  3,147,154,163, 73,166,244,255,255,255,255,
	148,117,182,255,255,255,255, 15, 56,148,181,103,255,255,255,  5, 81,  4,103,251,255,255,191,103, 56, 52, 69, 19,245,255, 89,164, 33,103,251,255,255,111,123, 33, 10, 56,148,245,255,103, 91,164,
	 36, 74, 32,255, 63,132, 83, 52, 82, 90,178,103, 39,115, 38, 69,249,255,255,159, 69,128,  6, 38,134,247,255, 99, 50,103, 81, 80,  4,255,111,130,134, 39,129,132, 21,133, 89,164, 97,113, 22,115,
	255, 31,166,113, 22,112,120,144, 69,  4, 74, 90, 48,106,122,115,122,166,167, 88,164,132,250,255,150,101,155,139,249,255,255, 63,182, 96,  3,101,144,245,255,176,  8,181, 16, 85,182,255,111, 59,
	 54, 85, 19,255,255,255, 33,154,181,185,184,101,255, 15, 59, 96, 11,105,101, 25,162,139,181,101,  8,165, 37, 32,101, 59, 54, 37, 58, 90,243,255,133, 89,130,101, 50, 40,255,159,101,105,  0, 38,
	255,255,255, 81, 24,  8,101, 56, 40, 38, 24,101, 18,246,255,255,255,255, 49, 22,166,131, 86,150,152,166,  1, 10,150,  5,101,240,255, 48, 88,166,255,255,255,255,175,101,255,255,255,255,255,255,
	 91,122,181,255,255,255,255,191,165,123,133,  3,255,255,255,181, 87,186,145,240,255,255,175, 87,186,151, 24, 56,241,255, 27,178, 23, 87,241,255,255, 15, 56, 33, 23, 87, 39,251,255,121,149,114,
	  9, 34,123,255,127, 37, 39, 91, 41, 35,152, 40, 82, 42, 83,115,245,255,255,143,  2, 88,130, 87, 42,245,255,  9, 81, 58, 53, 55, 42,255,159, 40, 41,129, 39, 42,117, 37, 49, 53, 87,255,255,255,
	255, 15,120,112, 17, 87,255,255,255,  9,147, 83, 53,247,255,255,159,120,149,247,255,255,255,255,133, 84,138,186,248,255,255, 95, 64,181, 80,186, 59,240,255, 16,137,164,168,171, 84,255,175, 75,
	 74,181, 67, 73, 49, 65, 82, 33, 88,178, 72,133,255, 15,180,176, 67,181,178, 81,177, 32,  5,149,178, 69,133,139,149, 84,178,243,255,255,255,255, 82, 58, 37, 67, 53, 72,255, 95, 42, 37, 68,  2,
	255,255,255,163, 50,165,131, 69,133, 16, 89, 42, 37, 20, 41, 73,242,255, 72,133, 53, 83,241,255,255, 15, 84,  1,245,255,255,255,255, 72,133, 53,  9,  5, 83,255,159, 84,255,255,255,255,255,255,
	180, 71,185,169,251,255,255, 15, 56,148,151,123,169,251,255,161, 27, 75, 65,112,180,255, 63, 65, 67, 24, 74, 71,171, 75,180,151, 75, 41,155, 33,255,159, 71,185,151,177,178,  1, 56,123,180, 36,
	 66,240,255,255,191, 71, 75,130, 67, 35,244,255,146, 42,151, 50,119,148,255,159,122,121,164,114,120, 32,112,115, 58, 42, 71, 26, 10,  4, 26, 42,120,244,255,255,255,255,148, 65,113, 23,243,255,
	255, 79, 25, 20,  7, 24,120,241,255,  4,115, 52,255,255,255,255, 79,120,255,255,255,255,255,255,169,168,139,255,255,255,255, 63,144,147,187,169,255,255,255, 16, 10,138,168,251,255,255, 63,161,
	 59,250,255,255,255,255, 33, 27,155,185,248,255,255, 63,144,147, 27,146,178,249,255, 32,139,176,255,255,255,255, 63,178,255,255,255,255,255,255, 50, 40,168,138,249,255,255,159, 42,144,242,255,
	255,255,255, 50, 40,168, 16, 24,138,255, 31, 42,255,255,255,255,255,255, 49,152,129,255,255,255,255, 15, 25,255,255,255,255,255,255, 48,248,255,255,255,255,255,255,255,255,255,255,255,255,255
};
)+R(uchar triangle_table(const uint i) {
	return (triangle_table_data[i/2u]>>(4u*(i%2u)))&0xF;
}
)+R(float interpolate(const float v1, const float v2, const float iso) { // linearly interpolate position where isosurface cuts an edge between 2 vertices at 0 and 1
	return (iso-v1)/(v2-v1);
}
)+R(uint marching_cubes(const float* v, const float iso, float3* triangles) { // input: 8 values v, isovalue; output: returns number of triangles, 15 triangle vertices t
	uint cube = 0u; // determine index of which vertices are inside of the isosurface
	for(uint i=0u; i<8u; i++) cube |= (v[i]<iso)<<i;
	if(cube==0u||cube==255u) return 0u; // cube is entirely inside/outside of the isosurface
	float3 vertex[12]; // find the vertices where the surface intersects the cube
	vertex[ 0] = (float3)(interpolate(v[0], v[1], iso), 0.0f, 0.0f); // interpolate vertices on all 12 edges
	vertex[ 1] = (float3)(1.0f, 0.0f, interpolate(v[1], v[2], iso)); // do interpolation only in 1D to reduce operations
	vertex[ 2] = (float3)(interpolate(v[3], v[2], iso), 0.0f, 1.0f);
	vertex[ 3] = (float3)(0.0f, 0.0f, interpolate(v[0], v[3], iso));
	vertex[ 4] = (float3)(interpolate(v[4], v[5], iso), 1.0f, 0.0f);
	vertex[ 5] = (float3)(1.0f, 1.0f, interpolate(v[5], v[6], iso));
	vertex[ 6] = (float3)(interpolate(v[7], v[6], iso), 1.0f, 1.0f);
	vertex[ 7] = (float3)(0.0f, 1.0f, interpolate(v[4], v[7], iso));
	vertex[ 8] = (float3)(0.0f, interpolate(v[0], v[4], iso), 0.0f);
	vertex[ 9] = (float3)(1.0f, interpolate(v[1], v[5], iso), 0.0f);
	vertex[10] = (float3)(1.0f, interpolate(v[2], v[6], iso), 1.0f);
	vertex[11] = (float3)(0.0f, interpolate(v[3], v[7], iso), 1.0f);
	cube *= 15u;
	uint i; // number of triangle vertices
	for(i=0u; i<15u; i+=3u) { // create the triangles
		const uchar triangle_table_cube_i_0u_ = triangle_table(cube+i);
		if(triangle_table_cube_i_0u_==15u) break;
		triangles[i   ] = vertex[triangle_table_cube_i_0u_];
		triangles[i+1u] = vertex[triangle_table(cube+i+1u)];
		triangles[i+2u] = vertex[triangle_table(cube+i+2u)];
	}
	return i/3u; // return number of triangles
}
)+R(uint marching_cubes_halfway(const bool* v, float3* triangles) { // input: 8 bool values v; output: returns number of triangles, 15 triangle vertices t
	uint cube = 0u; // determine index of which vertices are inside of the isosurface
	for(uint i=0u; i<8u; i++) cube |= (uint)(!v[i])<<i;
	if(cube==0u||cube==255u) return 0u; // cube is entirely inside/outside of the isosurface
	float3 vertex[12]; // find the vertices where the surface intersects the cube
	vertex[ 0] = (float3)(0.5f, 0.0f, 0.0f); // vertices on all 12 edges
	vertex[ 1] = (float3)(1.0f, 0.0f, 0.5f);
	vertex[ 2] = (float3)(0.5f, 0.0f, 1.0f);
	vertex[ 3] = (float3)(0.0f, 0.0f, 0.5f);
	vertex[ 4] = (float3)(0.5f, 1.0f, 0.0f);
	vertex[ 5] = (float3)(1.0f, 1.0f, 0.5f);
	vertex[ 6] = (float3)(0.5f, 1.0f, 1.0f);
	vertex[ 7] = (float3)(0.0f, 1.0f, 0.5f);
	vertex[ 8] = (float3)(0.0f, 0.5f, 0.0f);
	vertex[ 9] = (float3)(1.0f, 0.5f, 0.0f);
	vertex[10] = (float3)(1.0f, 0.5f, 1.0f);
	vertex[11] = (float3)(0.0f, 0.5f, 1.0f);
	cube *= 15u;
	uint i; // number of triangle vertices
	for(i=0u; i<15u; i+=3u) { // create the triangles
		const uchar triangle_table_cube_i_0u_ = triangle_table(cube+i);
		if(triangle_table_cube_i_0u_==15u) break;
		triangles[i   ] = vertex[triangle_table_cube_i_0u_];
		triangles[i+1u] = vertex[triangle_table(cube+i+1u)];
		triangles[i+2u] = vertex[triangle_table(cube+i+2u)];
	}
	return i/3u; // return number of triangles
}
)+R(typedef struct __attribute__((packed)) struct_ray {
	float3 origin;
	float3 direction;
} ray;
)+R(float intersect_sphere(const ray r, const float3 center, const float radius) {
	const float3 oc = center-r.origin;
	const float b=dot(oc, r.direction), c=sq(b)-dot(oc, oc)+sq(radius);
	return c<0.0f ? -1.0f : b-sqrt(c);
}
)+R(float intersect_sphere_inside(const ray r, const float3 center, const float radius) {
	const float3 oc = center-r.origin;
	const float b=dot(oc, r.direction), c=sq(b)-dot(oc, oc)+sq(radius);
	return c<0.0f ? -1.0f : b+sqrt(c);
}
)+R(float intersect_triangle(const ray r, const float3 p0, const float3 p1, const float3 p2) { // Moeller-Trumbore algorithm
	const float3 u=p1-p0, v=p2-p0, w=r.origin-p0, h=cross(r.direction, v), q=cross(w, u);
	const float g=dot(u, h), f=1.0f/g, s=f*dot(w, h), t=f*dot(r.direction, q);
	return (g<=0.0f||s<-0.0001f||s>1.0001f||t<-0.0001f||s+t>1.0001f) ? -1.0f : f*dot(v, q); // add tolerance values to avoid graphical artifacts with axis-aligned camera
}
)+R(float intersect_triangle_bidirectional(const ray r, const float3 p0, const float3 p1, const float3 p2) { // Moeller-Trumbore algorithm
	const float3 u=p1-p0, v=p2-p0, w=r.origin-p0, h=cross(r.direction, v), q=cross(w, u);
	const float g=dot(u, h), f=1.0f/g, s=f*dot(w, h), t=f*dot(r.direction, q);
	return (g==0.0f||s<-0.0001f||s>1.0001f||t<-0.0001f||s+t>1.0001f) ? -1.0f : f*dot(v, q); // add tolerance values to avoid graphical artifacts with axis-aligned camera
}
)+R(float intersect_rhombus(const ray r, const float3 p0, const float3 p1, const float3 p2) { // Moeller-Trumbore algorithm
	const float3 u=p1-p0, v=p2-p0, w=r.origin-p0, h=cross(r.direction, v), q=cross(w, u);
	const float g=dot(u, h), f=1.0f/g, s=f*dot(w, h), t=f*dot(r.direction, q);
	return (g<=0.0f||s<-0.0001f||s>1.0001f||t<-0.0001f||t>1.0001f) ? -1.0f : f*dot(v, q); // add tolerance values to avoid graphical artifacts with axis-aligned camera
}
)+R(float intersect_plane(const ray r, const float3 p0, const float3 p1, const float3 p2) { // ray-triangle intersection, but skip barycentric coordinates
	const float3 u=p1-p0, v=p2-p0, w=r.origin-p0, h=cross(r.direction, v);
	const float g = dot(u, h);
	return g<=0.0f ? -1.0f : dot(v, cross(w, u))/g;
}
)+R(float intersect_plane_bidirectional(const ray r, const float3 p0, const float3 p1, const float3 p2) { // ray-triangle intersection, but skip barycentric coordinates
	const float3 u=p1-p0, v=p2-p0, w=r.origin-p0, h=cross(r.direction, v);
	const float g = dot(u, h);
	return g==0.0f ? -1.0f : dot(v, cross(w, u))/g;
}
)+R(bool intersect_cuboid_bool(const ray r, const float3 center, const float Lx, const float Ly, const float Lz) {
	const float3 bmin = center-0.5f*(float3)(Lx, Ly, Lz);
	const float3 bmax = center+0.5f*(float3)(Lx, Ly, Lz);
	const float txa = (bmin.x-r.origin.x)/r.direction.x;
	const float txb = (bmax.x-r.origin.x)/r.direction.x;
	const float txmin = fmin(txa, txb);
	const float txmax = fmax(txa, txb);
	const float tya = (bmin.y-r.origin.y)/r.direction.y;
	const float tyb = (bmax.y-r.origin.y)/r.direction.y;
	const float tymin = fmin(tya, tyb);
	const float tymax = fmax(tya, tyb);
	if(txmin>tymax||tymin>txmax) return false;
	const float tza = (bmin.z-r.origin.z)/r.direction.z;
	const float tzb = (bmax.z-r.origin.z)/r.direction.z;
	const float tzmin = fmin(tza, tzb);
	const float tzmax = fmax(tza, tzb);
	return fmax(txmin, tymin)<=tzmax&&tzmin<=fmin(txmax, tymax);
}
)+R(float intersect_cuboid(const ray r, const float3 center, const float Lx, const float Ly, const float Lz) {
	const float3 bmin = center-0.5f*(float3)(Lx, Ly, Lz);
	const float3 bmax = center+0.5f*(float3)(Lx, Ly, Lz);
	if(r.origin.x>=bmin.x&&r.origin.y>=bmin.y&&r.origin.z>=bmin.z&&r.origin.x<=bmax.x&&r.origin.y<=bmax.y&&r.origin.z<=bmax.z) return 0.0f; // ray origin is within cuboid
	float3 p[8]; // 8 cuboid vertices
	p[0] = (float3)(bmin.x, bmin.y, bmin.z); // ---
	p[1] = (float3)(bmax.x, bmin.y, bmin.z); // +--
	p[2] = (float3)(bmax.x, bmin.y, bmax.z); // +-+
	p[3] = (float3)(bmin.x, bmin.y, bmax.z); // --+
	p[4] = (float3)(bmin.x, bmax.y, bmin.z); // -+-
	p[5] = (float3)(bmax.x, bmax.y, bmin.z); // ++-
	p[6] = (float3)(bmax.x, bmax.y, bmax.z); // +++
	p[7] = (float3)(bmin.x, bmax.y, bmax.z); // -++
	float intersect = -1.0f; // test for intersections with the 6 cuboid faces, ray will intersect with either 0 or 1 rhombuses
	intersect = fmax(intersect, intersect_rhombus(r, p[0], p[3], p[4])); // +00 (normal vectors)
	intersect = fmax(intersect, intersect_rhombus(r, p[2], p[1], p[6])); // -00
	intersect = fmax(intersect, intersect_rhombus(r, p[1], p[2], p[0])); // 0+0
	intersect = fmax(intersect, intersect_rhombus(r, p[7], p[6], p[4])); // 0-0
	intersect = fmax(intersect, intersect_rhombus(r, p[1], p[0], p[5])); // 00+
	intersect = fmax(intersect, intersect_rhombus(r, p[3], p[2], p[7])); // 00-
	return intersect;
}
)+R(float intersect_cuboid_inside_with_normal(const ray r, const float3 center, const float Lx, const float Ly, const float Lz, float3* normal) {
	const float3 bmin = center-0.5f*(float3)(Lx, Ly, Lz);
	const float3 bmax = center+0.5f*(float3)(Lx, Ly, Lz);
	float3 p[8]; // 8 cuboid vertices
	p[0] = (float3)(bmin.x, bmin.y, bmin.z); // ---
	p[1] = (float3)(bmax.x, bmin.y, bmin.z); // +--
	p[2] = (float3)(bmax.x, bmin.y, bmax.z); // +-+
	p[3] = (float3)(bmin.x, bmin.y, bmax.z); // --+
	p[4] = (float3)(bmin.x, bmax.y, bmin.z); // -+-
	p[5] = (float3)(bmax.x, bmax.y, bmin.z); // ++-
	p[6] = (float3)(bmax.x, bmax.y, bmax.z); // +++
	p[7] = (float3)(bmin.x, bmax.y, bmax.z); // -++
	float intersect = -1.0f; // test for intersections with the 6 cuboid faces
	float rhombus_intersect[6];
	rhombus_intersect[0] = intersect_rhombus(r, p[2], p[6], p[1]); // +00 (normal vectors, points 2 and 3 are switched here to flip rhombus around)
	rhombus_intersect[1] = intersect_rhombus(r, p[0], p[4], p[3]); // -00
	rhombus_intersect[2] = intersect_rhombus(r, p[7], p[4], p[6]); // 0+0
	rhombus_intersect[3] = intersect_rhombus(r, p[1], p[0], p[2]); // 0-0
	rhombus_intersect[4] = intersect_rhombus(r, p[3], p[7], p[2]); // 00+
	rhombus_intersect[5] = intersect_rhombus(r, p[1], p[5], p[0]); // 00-
	uint side = 0u; // test for intersections with the 6 cuboid faces
	for(uint i=0u; i<6u; i++) {
		if(rhombus_intersect[i]>intersect) { // test for intersections with the 6 cuboid faces
			intersect = rhombus_intersect[i]; // ray will intersect with either 0 or 1 rhombuses
			side = i;
		}
	}
	*normal = (float3)(side==0u ? 1.0f : side==1u ? -1.0f : 0.0f, side==2u ? 1.0f : side==3u ? -1.0f : 0.0f, side==4u ? 1.0f : side==5u ? -1.0f : 0.0f);
	return intersect;
}
)+R(float3 reflect(const float3 direction, const float3 normal) {
	return direction-2.0f*dot(direction, normal)*normal;
}
)+R(float3 refract(const float3 direction, const float3 normal, const float n) {
	const float direction_normal = dot(direction, normal);
	const float sqrt_part = sq(n)-1.0f+sq(direction_normal);
	return sqrt_part>=0.0f ? (direction-(direction_normal+sqrt(sqrt_part))*normal)/n : direction-2.0f*direction_normal*normal; // refraction : total internal reflection
}
)+R(ray get_camray(const int x, const int y, const float* camera_cache) {
	const float zoom = camera_cache[0]; // fetch camera parameters (rotation matrix, camera position, etc.)
	const float  dis = camera_cache[1];
	const float3 pos = (float3)(camera_cache[ 2], camera_cache[ 3], camera_cache[ 4])-(float3)(def_domain_offset_x, def_domain_offset_y, def_domain_offset_z);
	const float3 Rx  = (float3)(camera_cache[ 5], camera_cache[ 6], camera_cache[ 7]);
	const float3 Ry  = (float3)(camera_cache[ 8], camera_cache[ 9], camera_cache[10]);
	const float3 Rz  = (float3)(camera_cache[11], camera_cache[12], camera_cache[13]);
	const bool   vr  = (as_int(camera_cache[14])>>31)&0x1;
	const float  rtv = (as_int(camera_cache[14])>>30)&0x1 ? 2.0f : 1.0f;
	const float  eye_distance = vload_half(28, (half*)camera_cache);
	const float  stereo = (x<(int)def_screen_width/2 ? -1.0f : 1.0f);
	float3 p0 = (float3)(!vr ? 0.0f : stereo*eye_distance/zoom, 0.0f, dis/zoom);
	float3 p1 = p0+normalize((float3)(!vr ? (float)(x-(int)def_screen_width/2) : ((float)(x-(int)def_screen_width/2)-stereo*(float)(def_screen_width/4u))*rtv-stereo*eye_distance, (float)(y-(int)def_screen_height/2), -dis));
	p0 = Rx*p0.x+Ry*p0.y+Rz*p0.z+pos; // reverse rotation and reverse transformation of p0
	p1 = Rx*p1.x+Ry*p1.y+Rz*p1.z+pos; // reverse rotation and reverse transformation of p1
	ray camray;
	camray.origin = p0;
	camray.direction = p1-p0;
	return camray;
}
)+R(int skybox_bottom(const ray r, const int c1, const int c2, const int skybox_color) {
	const float3 p0=(float3)(0.0f, 0.0f, -0.5f*(float)def_Nz), p1=(float3)(1.0f, 0.0f, -0.5f*(float)def_Nz), p2=(float3)(0.0f, 1.0f, -0.5f*(float)def_Nz);
	const float distance = intersect_plane(r, p0, p1, p2);
	if(distance>0.0f) { // ray intersects with bottom
		const float3 normal = normalize(cross(p1-p0, p2-p0));
		float3 intersection = r.origin+distance*r.direction;
		const float scale = 2.0f/fmin((float)def_Nx, (float)def_Ny);
		int a = abs((int)floor(scale*intersection.x));
		int b = abs((int)floor(scale*intersection.y));
		const float r = scale*sqrt(sq(intersection.x)+sq(intersection.y));
		const int w = (a%2==b%2);
		return color_mix(w*c1+(1-w)*c2, color_mix(c1, c2, 0.5f), clamp(10.0f/r, 0.0f, 1.0f));
	} else {
		return skybox_color;
	}
}
)+R(int skybox_color_bw(const float x, const float y) {
	return color_mul(0xFFFFFF, 1.0f-y);
}
)+R(int skybox_color_hsv(const float x, const float y) {
	const float h = fmod(x*360.0f+120.0f, 360.0f);
	const float s = y>0.5f ? 1.0f : 2.0f*y;
	const float v = y>0.5f ? 2.0f-2.0f*y : 1.0f;
	return hsv_to_rgb(h, s, v);
}
)+R(int skybox_color_sunset(const float x, const float y) {
	return color_mix(255<<16|175<<8|55, y<0.5f ? 55<<16|111<<8|255 : 0, 2.0f*(0.5f-fabs(y-0.5f)));
}
)+R(int skybox_color_grid(const float x, const float y, const int c1, const int c2) {
	int a = (int)(72.0f*x);
	int b = (int)(36.0f*y);
	const int w = (a%2==b%2);
	return w*c1+(1-w)*c2;
}
)+R(int skybox_color(const ray r, const global int* skybox) {
	const float3 direction = normalize(r.direction); // to avoid artifacts from asin(direction.z)
	//const float x = fma(atan2(direction.x, direction.y),  0.5f/3.1415927f, 0.5f);
	//const float y = fma(asin (direction.z             ), -1.0f/3.1415927f, 0.5f);
	//return skybox_color_bw(x, y);
	//return color_mix(skybox_color_hsv(x, y), skybox_color_grid(x, y, 0xFFFFFF, 0x000000), 0.95f-0.33f*(2.0f*(0.5f-fabs(y-0.5f))));
	//return skybox_bottom(r, 0xFFFFFF, 0xF0F0F0, skybox_color_grid(x, y, 0xFFFFFF, 0xF0F0F0));
	const float fu = (float)def_skybox_width *fma(atan2(direction.x, direction.y),  0.5f/3.1415927f, 0.5f);
	const float fv = (float)def_skybox_height*fma(asin (direction.z             ), -1.0f/3.1415927f, 0.5f);
	const int ua=clamp((int)fu, 0, (int)def_skybox_width-1), va=clamp((int)fv, 0, (int)def_skybox_height-1), ub=(ua+1)%def_skybox_width, vb=min(va+1, (int)def_skybox_height-1); // bilinear interpolation positions
	const int s00=skybox[ua+va*def_skybox_width], s01=skybox[ua+vb*def_skybox_width], s10=skybox[ub+va*def_skybox_width], s11=skybox[ub+vb*def_skybox_width];
	const float u1=fu-(float)ua, v1=fv-(float)va, u0=1.0f-u1, v0=1.0f-v1; // interpolation factors
	return color_mix(color_mix(s00, s01, v0), color_mix(s10, s11, v0), u0); // perform bilinear interpolation
}
)+R(int last_ray(const ray reflection, const ray transmission, const float reflectivity, const float transmissivity, const global int* skybox) {
	return color_mix(skybox_color(reflection, skybox), color_mix(skybox_color(transmission, skybox), def_absorption_color, transmissivity), reflectivity);
}
)+R(float interpolate_phi(const float3 p, const global float* phi, const uint Nx, const uint Ny, const uint Nz) { // trilinear interpolation of velocity at point p
	const float xa=p.x-0.5f+1.5f*(float)Nx, ya=p.y-0.5f+1.5f*(float)Ny, za=p.z-0.5f+1.5f*(float)Nz; // subtract lattice offsets
	const uint xb=(uint)xa, yb=(uint)ya, zb=(uint)za; // integer casting to find bottom left corner
	const float x1=xa-(float)xb, y1=ya-(float)yb, z1=za-(float)zb, x0=1.0f-x1, y0=1.0f-y1, z0=1.0f-z1; // calculate interpolation factors
	float phin[8]; // phi at unit cube corner points
	for(uint c=0u; c<8u; c++) { // count over eight corner points
		const uint i=(c&0x04u)>>2, j=(c&0x02u)>>1, k=c&0x01u; // disassemble c into corner indices ijk
		const uint x=(xb+i)%Nx, y=(yb+j)%Ny, z=(zb+k)%Nz; // calculate corner lattice positions
		const uxx n = (uxx)x+(uxx)(y+z*Ny)*(uxx)Nx; // calculate lattice linear index
		phin[c] = phi[n]; // load velocity from lattice point
	}
	return (x0*y0*z0)*phin[0]+(x0*y0*z1)*phin[1]+(x0*y1*z0)*phin[2]+(x0*y1*z1)*phin[3]+(x1*y0*z0)*phin[4]+(x1*y0*z1)*phin[5]+(x1*y1*z0)*phin[6]+(x1*y1*z1)*phin[7]; // perform trilinear interpolation
}
)+R(float ray_grid_traverse(const ray r, const global float* phi, const global uchar* flags, float3* normal, const uint Nx, const uint Ny, const uint Nz) {
	const float3 p = (float3)(r.origin.x-0.5f+0.5f*(float)Nx, r.origin.y-0.5f+0.5f*(float)Ny, r.origin.z-0.5f+0.5f*(float)Nz); // start point
	const int dx=(int)sign(r.direction.x), dy=(int)sign(r.direction.y), dz=(int)sign(r.direction.z); // fast ray-grid-traversal
	int3 xyz = (int3)((int)floor(p.x), (int)floor(p.y), (int)floor(p.z));
	const float fxa=p.x-floor(p.x), fya=p.y-floor(p.y), fza=p.z-floor(p.z);
	const float tdx = 1.0f/fmax(fabs(r.direction.x), 1E-6f);
	const float tdy = 1.0f/fmax(fabs(r.direction.y), 1E-6f);
	const float tdz = 1.0f/fmax(fabs(r.direction.z), 1E-6f);
	float tmx = tdx*(dx>0 ? 1.0f-fxa : dx<0 ? fxa : 0.0f);
	float tmy = tdy*(dy>0 ? 1.0f-fya : dy<0 ? fya : 0.0f);
	float tmz = tdz*(dz>0 ? 1.0f-fza : dz<0 ? fza : 0.0f);
	for(uint tc=0u; tc<Nx+Ny+Nz; tc++) { // limit number of traversed cells to space diagonal
		if(tmx<tmy) { if(tmx<tmz) { xyz.x += dx; tmx += tdx; } else { xyz.z += dz; tmz += tdz; } }
		else /****/ { if(tmy<tmz) { xyz.y += dy; tmy += tdy; } else { xyz.z += dz; tmz += tdz; } }
		if(xyz.x<-1 || xyz.y<-1 || xyz.z<-1 || xyz.x>=(int)Nx || xyz.y>=(int)Ny || xyz.z>=(int)Nz) break; // out of simulation box
		else if(xyz.x<0 || xyz.y<0 || xyz.z<0 || xyz.x>=(int)Nx-1 || xyz.y>=(int)Ny-1 || xyz.z>=(int)Nz-1) continue;
		const uxx x0 = (uxx)        xyz.x; // cube stencil
		const uxx xp = (uxx) (      xyz.x+1);
		const uxx y0 = (uxx)( (uint)xyz.y   *Nx);
		const uxx yp = (uxx)(((uint)xyz.y+1)*Nx);
		const uxx z0 = (uxx)        xyz.z   *(uxx)(Ny*Nx);
		const uxx zp = (uxx) (      xyz.z+1)*(uxx)(Ny*Nx);
		uxx j[8];
		j[0] = x0+y0+z0; // 000
		j[1] = xp+y0+z0; // +00
		j[2] = xp+y0+zp; // +0+
		j[3] = x0+y0+zp; // 00+
		j[4] = x0+yp+z0; // 0+0
		j[5] = xp+yp+z0; // ++0
		j[6] = xp+yp+zp; // +++
		j[7] = x0+yp+zp; // 0++
		uchar flags_cell = 0u; // check with cheap flags if the isosurface goes through the current marching-cubes cell (~15% performance boost)
		for(uint i=0u; i<8u; i++) flags_cell |= flags[j[i]];
		if(!(flags_cell&(TYPE_S|TYPE_E|TYPE_I))) continue; // cell is entirely inside/outside of the isosurface
		float v[8];
		for(uint i=0u; i<8u; i++) v[i] = phi[j[i]];
		float3 triangles[15]; // maximum of 5 triangles with 3 vertices each
		const uint tn = marching_cubes(v, 0.5f, triangles); // run marching cubes algorithm
		if(tn==0u) continue; // if returned tn value is non-zero, iterate through triangles
		const float3 offset = (float3)((float)xyz.x+0.5f-0.5f*(float)Nx, (float)xyz.y+0.5f-0.5f*(float)Ny, (float)xyz.z+0.5f-0.5f*(float)Nz);
		for(uint i=0u; i<tn; i++) {
			const float3 p0 = triangles[3u*i   ]+offset;
			const float3 p1 = triangles[3u*i+1u]+offset;
			const float3 p2 = triangles[3u*i+2u]+offset;
			const float intersect = intersect_triangle_bidirectional(r, p0, p1, p2); // for each triangle, check ray-triangle intersection
			if(intersect>0.0f) { // intersection found (there can only be exactly 1 intersection)
				const uxx xq = (uxx) (((uint)xyz.x   +2u)%Nx); // central difference stencil on each cube corner point
				const uxx xm = (uxx) (((uint)xyz.x+Nx-1u)%Nx);
				const uxx yq = (uxx)((((uint)xyz.y   +2u)%Ny)*Nx);
				const uxx ym = (uxx)((((uint)xyz.y+Ny-1u)%Ny)*Nx);
				const uxx zq = (uxx) (((uint)xyz.z   +2u)%Nz)*(uxx)(Ny*Nx);
				const uxx zm = (uxx) (((uint)xyz.z+Nz-1u)%Nz)*(uxx)(Ny*Nx);
				float3 n[8];
				n[0] = (float3)(phi[xm+y0+z0]-v[1], phi[x0+ym+z0]-v[4], phi[x0+y0+zm]-v[3]); // central difference stencil on each cube corner point
				n[1] = (float3)(v[0]-phi[xq+y0+z0], phi[xp+ym+z0]-v[5], phi[xp+y0+zm]-v[2]); // compute normal vectors from gradient
				n[2] = (float3)(v[3]-phi[xq+y0+zp], phi[xp+ym+zp]-v[6], v[1]-phi[xp+y0+zq]); // normalize later after trilinear interpolation
				n[3] = (float3)(phi[xm+y0+zp]-v[2], phi[x0+ym+zp]-v[7], v[0]-phi[x0+y0+zq]);
				n[4] = (float3)(phi[xm+yp+z0]-v[5], v[0]-phi[x0+yq+z0], phi[x0+yp+zm]-v[7]);
				n[5] = (float3)(v[4]-phi[xq+yp+z0], v[1]-phi[xp+yq+z0], phi[xp+yp+zm]-v[6]);
				n[6] = (float3)(v[7]-phi[xq+yp+zp], v[2]-phi[xp+yq+zp], v[5]-phi[xp+yp+zq]);
				n[7] = (float3)(phi[xm+yp+zp]-v[6], v[3]-phi[x0+yq+zp], v[4]-phi[x0+yp+zq]);
				const float3 p = r.origin+intersect*r.direction-offset; // intersection point minus offset
				*normal = normalize(trilinear3(p-floor(p), n)); // perform trilinear interpolation and normalization
				return intersect; // intersection found, exit loop
			}
		}
	}
	const float intersect = intersect_cuboid_inside_with_normal(r, (float3)(0.0f, 0.0f, 0.0f), (float)Nx-1.0f, (float)Ny-1.0f, (float)Nz-1.0f, normal); // -1 because marching-cubes surface ends between cells
	return intersect>0.0f ? (interpolate_phi(r.origin+intersect*r.direction, phi, Nx, Ny, Nz)>0.5f ? intersect : -1.0f) : -1.0f; // interpolate phi at ray intersection point with simulation box, to check if ray is inside fluid
}
)+R(bool raytrace_phi_mirror(const ray ray_in, ray* ray_reflect, const global float* phi, const global uchar* flags, const global int* skybox, const uint Nx, const uint Ny, const uint Nz) { // only reflection
	float3 normal;
	float d = ray_grid_traverse(ray_in, phi, flags, &normal, Nx, Ny, Nz); // move ray through lattice, at each cell call marching_cubes
	if(d==-1.0f) return false; // no intersection found
	ray_reflect->origin = ray_in.origin+(d-0.005f)*ray_in.direction; // start intersection points a bit in front triangle to avoid self-reflection
	ray_reflect->direction = reflect(ray_in.direction, normal);
	return true;
}
)+R(bool raytrace_phi(const ray ray_in, ray* ray_reflect, ray* ray_transmit, float* reflectivity, float* transmissivity, const global float* phi, const global uchar* flags, const global int* skybox, const uint Nx, const uint Ny, const uint Nz) {
	float3 normal;
	float d = ray_grid_traverse(ray_in, phi, flags, &normal, Nx, Ny, Nz); // move ray through lattice, at each cell call marching_cubes
	if(d==-1.0f) return false; // no intersection found
	const float ray_in_normal = dot(ray_in.direction, normal);
	const bool is_inside = ray_in_normal>0.0f; // camera is in fluid
	ray_reflect->origin = ray_in.origin+(d-0.005f)*ray_in.direction; // start intersection points a bit in front triangle to avoid self-reflection
	ray_reflect->direction = reflect(ray_in.direction, normal); // compute reflection ray
	ray ray_internal; // compute internal ray and transmission ray
	ray_internal.origin = ray_in.origin+(d+0.005f)*ray_in.direction; // start intersection points a bit behind triangle to avoid self-transmission
	ray_internal.direction = refract(ray_in.direction, normal, def_n);
	const float wr = clamp(sq(cb(2.0f*acospi(fabs(ray_in_normal)))), 0.0f, 1.0f); // increase reflectivity if ray intersects surface at shallow angle
	if(is_inside) { // swap ray_reflect and ray_internal
		const float3 ray_internal_origin = ray_internal.origin;
		ray_internal.origin = ray_reflect->origin;
		ray_internal.direction = ray_reflect->direction;
		ray_reflect->origin = ray_internal_origin; // re-use internal ray origin
		ray_reflect->direction = refract(ray_in.direction, -normal, 1.0f/def_n); // compute refraction again: refract out of fluid
		if(sq(1.0f/def_n)-1.0f+sq(ray_in_normal)>=0.0f) { // refraction through Snell's window
			ray_transmit->origin = ray_reflect->origin; // reflection ray and transmission ray are the same
			ray_transmit->direction = ray_reflect->direction;
			*reflectivity = 0.0f;
			*transmissivity = exp(def_attenuation*d); // Beer-Lambert law
			return true;
		}
	}
	float d_internal = d;
	d = ray_grid_traverse(ray_internal, phi, flags, &normal, Nx, Ny, Nz); // 2nd ray-grid traversal call: refraction (camera outside) or total internal reflection (camera inside)
	ray_transmit->origin = d!=-1.0f ? ray_internal.origin+(d+0.005f)*ray_internal.direction : ray_internal.origin; // start intersection points a bit behind triangle to avoid self-transmission
	ray_transmit->direction = d!=-1.0f||is_inside ? refract(ray_internal.direction, -normal, 1.0f/def_n) : ray_internal.direction; // internal ray intersects isosurface : internal ray does not intersect again
	*reflectivity = is_inside ? 0.0f : wr; // is_inside means camera is inside fluid, so this is a total internal reflection down here
	*transmissivity = d!=-1.0f||is_inside ? exp(def_attenuation*((float)is_inside*d_internal+d)) : (float)(def_attenuation==0.0f); // Beer-Lambert law
	return true;
}
)+R(bool is_above_plane(const float3 point, const float3 plane_p, const float3 plane_n) {
	return dot(point-plane_p, plane_n)>=0.0f;
}
)+R(bool is_below_plane(const float3 point, const float3 plane_p, const float3 plane_n) {
	return dot(point-plane_p, plane_n)<=0.0f;
}
)+R(bool is_in_camera_frustum(const float3 p, const float* camera_cache) { // returns true if point is located in camera frustum
	const float zoom = camera_cache[0]; // fetch camera parameters (rotation matrix, camera position, etc.)
	const float  dis = camera_cache[1];
	const float3 pos = (float3)(camera_cache[ 2], camera_cache[ 3], camera_cache[ 4])-(float3)(def_domain_offset_x, def_domain_offset_y, def_domain_offset_z);
	const float3 Rx  = (float3)(camera_cache[ 5], camera_cache[ 6], camera_cache[ 7]);
	const float3 Ry  = (float3)(camera_cache[ 8], camera_cache[ 9], camera_cache[10]);
	const float3 Rz  = (float3)(camera_cache[11], camera_cache[12], camera_cache[13]);
	const bool   vr  = (as_int(camera_cache[14])>>31)&0x1;
	const float  rtv = (as_int(camera_cache[14])>>30)&0x1 ? 2.0f : 1.0f;
	const float3 p0 = (float3)(0.0f, 0.0f, dis/zoom);
	const float3 camera_center = Rx*p0.x+Ry*p0.y+Rz*p0.z+pos; // reverse rotation and reverse transformation of p0
	const float x_left   = !vr ? (float)(-(int)def_screen_width/2  ) : ((float)(-(int)def_screen_width/2  )+(float)(def_screen_width/4u))*rtv;
	const float x_right  = !vr ? (float)( (int)def_screen_width/2-1) : ((float)( (int)def_screen_width/2-1)-(float)(def_screen_width/4u))*rtv;
	const float y_top    = (float)(-(int)def_screen_height/2 );
	const float y_bottom = (float)((int)def_screen_height/2-1);
	const float dis_clamped = fmin(dis, 1E4f); // avoid flickering at very small field of view
	float3 r00 = p0+normalize((float3)(x_left , y_top   , -dis_clamped)); // get 4 edge vectors of frustum, get_camray(...) inlined and redundant parts eliminated
	float3 r01 = p0+normalize((float3)(x_right, y_top   , -dis_clamped));
	float3 r10 = p0+normalize((float3)(x_left , y_bottom, -dis_clamped));
	float3 r11 = p0+normalize((float3)(x_right, y_bottom, -dis_clamped));
	r00 = Rx*r00.x+Ry*r00.y+Rz*r00.z+pos-camera_center; // reverse rotation and reverse transformation of r00
	r01 = Rx*r01.x+Ry*r01.y+Rz*r01.z+pos-camera_center; // reverse rotation and reverse transformation of r01
	r10 = Rx*r10.x+Ry*r10.y+Rz*r10.z+pos-camera_center; // reverse rotation and reverse transformation of r10
	r11 = Rx*r11.x+Ry*r11.y+Rz*r11.z+pos-camera_center; // reverse rotation and reverse transformation of r11
	const float3 plane_n_top    = cross(r00, r01); // get 4 frustum planes
	const float3 plane_n_bottom = cross(r11, r10);
	const float3 plane_n_left   = cross(r10, r00);
	const float3 plane_n_right  = cross(r01, r11);
	const float3 plane_p_top    = camera_center-2.0f*plane_n_top; // move frustum planes outward by 2 units
	const float3 plane_p_bottom = camera_center-2.0f*plane_n_bottom;
	const float3 plane_p_left   = camera_center-(2.0f+8.0f*(float)vr)*plane_n_left; // move frustum planes outward by 2 units, for stereoscopic rendering a bit more
	const float3 plane_p_right  = camera_center-(2.0f+8.0f*(float)vr)*plane_n_right;
	return is_above_plane(p, plane_p_top, plane_n_top)&&is_above_plane(p, plane_p_bottom, plane_n_bottom)&&is_above_plane(p, plane_p_left, plane_n_left)&&is_above_plane(p, plane_p_right, plane_n_right);
}
)+"#endif"+R( // GRAPHICS



// ################################################## LBM code ##################################################

)+R(uint3 coordinates(const uxx n) { // disassemble 1D index to 3D coordinates (n -> x,y,z)
	const uint t = (uint)(n%(uxx)(def_Nx*def_Ny));
	return (uint3)(t%def_Nx, t/def_Nx, (uint)(n/(uxx)(def_Nx*def_Ny))); // n = x+(y+z*Ny)*Nx
}
)+R(uxx index(const uint3 xyz) { // assemble 1D index from 3D coordinates (x,y,z -> n)
	return (uxx)xyz.x+(uxx)(xyz.y+xyz.z*def_Ny)*(uxx)def_Nx; // n = x+(y+z*Ny)*Nx
}
)+"#ifdef RHO_RAND"+R(
)+R(uxx rr_idx(const uxx n) { // ★ 15.09.2026 RHO_RAND C2b: Zelle -> Index in der Randschale R1 (Dicke 2, RHO_RAND-PLAN.md §4); def_RR_N = nicht in R1 (Papierkorb-Slot)
	const uint3 c = coordinates(n);
	const uxx a = (uxx)def_Nx*(uxx)def_Ny;
	if(c.z<2u) return n;
	if(c.z+2u>=def_Nz) return 2u*a+(n-(uxx)(def_Nz-2u)*a);
	const uxx r0 = 4u*a+(uxx)(c.z-2u)*(uxx)(4u*def_Nx+4u*(def_Ny-4u));
	if(c.y<2u) return r0+(uxx)c.x+(uxx)c.y*(uxx)def_Nx;
	if(c.y+2u>=def_Ny) return r0+2u*(uxx)def_Nx+(uxx)c.x+(uxx)(c.y+2u-def_Ny)*(uxx)def_Nx;
	if(c.x<2u) return r0+4u*(uxx)def_Nx+4u*(uxx)(c.y-2u)+(uxx)c.x;
	if(c.x+2u>=def_Nx) return r0+4u*(uxx)def_Nx+4u*(uxx)(c.y-2u)+(uxx)(c.x+4u-def_Nx);
	return (uxx)def_RR_N;
}
)+"#endif"+R( // RHO_RAND
)+R(float3 position(const uint3 xyz) { // 3D coordinates to 3D position
	return (float3)((float)xyz.x+0.5f-0.5f*(float)def_Nx, (float)xyz.y+0.5f-0.5f*(float)def_Ny, (float)xyz.z+0.5f-0.5f*(float)def_Nz);
}
)+R(uint3 closest_coordinates(const float3 p) { // return closest lattice point to point p
	return (uint3)((uint)(p.x+1.5f*(float)def_Nx)%def_Nx, (uint)(p.y+1.5f*(float)def_Ny)%def_Ny, (uint)(p.z+1.5f*(float)def_Nz)%def_Nz);
}
)+R(float3 mirror_position(const float3 p) { // mirror position into periodic boundaries
	float3 r;
	r.x = sign(p.x)*(fmod(fabs(p.x)+0.5f*(float)def_GNx, (float)def_GNx)-0.5f*(float)def_GNx);
	r.y = sign(p.y)*(fmod(fabs(p.y)+0.5f*(float)def_GNy, (float)def_GNy)-0.5f*(float)def_GNy);
	r.z = sign(p.z)*(fmod(fabs(p.z)+0.5f*(float)def_GNz, (float)def_GNz)-0.5f*(float)def_GNz);
	return r;
}
)+R(float3 mirror_distance(const float3 d) { // mirror distance vector into periodic boundaries
	return mirror_position(d);
}
)+R(bool is_halo(const uxx n) {
	const uint3 xyz = coordinates(n);
	return ((def_Dx>1u)&(xyz.x==0u||xyz.x>=def_Nx-1u))||((def_Dy>1u)&(xyz.y==0u||xyz.y>=def_Ny-1u))||((def_Dz>1u)&(xyz.z==0u||xyz.z>=def_Nz-1u));
}

)+R(float half_to_float_custom(const ushort x) { // custom 16-bit floating-point format, 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits
	const uint e = (x&0x7800)>>11; // exponent
	const uint m = (x&0x07FF)<<12; // mantissa
	const uint v = as_uint((float)m)>>23; // evil log2 bit hack to count leading zeros in denormalized format
	return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FF000))); // sign : normalized : denormalized
}
)+R(ushort float_to_half_custom(const float x) { // custom 16-bit floating-point format, 1-4-11, exp-15, +-1.99951168, +-6.10351562E-5, +-2.98023224E-8, 3.612 digits
	const uint b = as_uint(x)+0x00000800; // round-to-nearest-even: add last bit after truncated mantissa
	const uint e = (b&0x7F800000)>>23; // exponent
	const uint m = b&0x007FFFFF; // mantissa; in line below: 0x007FF800 = 0x00800000-0x00000800 = decimal indicator flag - initial rounding
	return (b&0x80000000)>>16 | (e>112)*((((e-112)<<11)&0x7800)|m>>12) | ((e<113)&(e>100))*((((0x007FF800+m)>>(124-e))+1)>>1); // sign : normalized : denormalized (assume [-2,2])
}

)+"#ifdef FORCE_FIELD"+R(
// FORK -- F-Bounding-Box. F wird nur um den Koerper herum alloziert statt ueber die ganze Domaene.
// Ohne das braucht der Fahrzeugfall bei 4 mm rund 6 GB allein fuer F und passt nicht mehr in die
// 32 GB der B70; mit ihm laeuft er bei 29,6 GB (im alten Baum gemessen, nicht geschaetzt).
// Ausserhalb der Box liefert load3_F null und store3_F verwirft -- die Kraft interessiert dort nicht.
// Bei voller Domaene ist def_FBN == def_N und der Index identisch, also bit-identisch zu Upstream.
bool f_bbox(const uxx n, uxx* fbi) {
	const uint3 xyz = coordinates(n);
	if(xyz.x<def_FBX0||xyz.x>=def_FBX0+def_FBNX||xyz.y<def_FBY0||xyz.y>=def_FBY0+def_FBNY||xyz.z<def_FBZ0||xyz.z>=def_FBZ0+def_FBNZ) return false;
	*fbi = (uxx)(xyz.x-def_FBX0)+(uxx)(xyz.y-def_FBY0)*def_FBNX+(uxx)(xyz.z-def_FBZ0)*(uxx)def_FBNX*(uxx)def_FBNY;
	return true;
}
// FORK 03.09.2026 -- F-MARKERLISTE (JIT-Define F_LISTE, Laufzeitschalter CFD_F_LISTE, Default AUS).
// GEMESSEN am 8-mm-Fahrzeug (CFD_F_LISTE_ZENSUS): von 20.805.960 F-BBox-Zellen sind 8.233.611 solid,
// aber nur 849.790 davon WANDsolid (>=1 Nicht-Solid unter den 18 D3Q19-Links) -- 4,08 % der Box.
// Genau diese Zellen schreibt update_force_field; alle anderen tragen konstruktiv F=0. F als volles
// BBox-Feld kostet deshalb am 4-mm-Fahrzeug 1.832 MiB fuer eine Belegung von rund zwei Prozent.
// Mit Liste: 3 float je Wandsolidzelle + dieselbe Bitmasken/Praefix-Maschinerie wie fac_idx.
//
// DIE MASKE IST BEWUSST EINE OBERMENGE. Der Host baut sie (alloc_f_liste) und wickelt x/y periodisch,
// waehrend der Kernel neighbors() in ALLEN Richtungen wickelt. Eine Zelle, die der Kernel schreibt,
// aber der Host nicht in der Maske haette, waere ein STILLER Kraftverlust -- deshalb nimmt der Host
// im Zweifel mehr auf (ein paar ungenutzte Slots kosten nichts). Die Gegenrichtung faengt der
// Wirkpfad-Zaehler: jeder store3_F ohne Slot zaehlt Slot 77 hoch, Soll ist 0 am Laufende.
//
// Der Maskenpuffer ist IMMER Parameter dieser Funktionen, auch wenn F_LISTE aus ist -- dann wird er
// nicht gelesen. Das haelt die Parameterlisten der fuenf F-Kernel ueber beide Arme konstant und
// vermeidet die Signaturversatz-Fehlerklasse, die dieses Projekt am 02.09. zweimal bezahlt hat.
bool f_slot(const global uint* f_maske, const uxx n, uxx* slot) {
	uxx fbi; if(!f_bbox(n, &fbi)) return false;
)+"#ifdef F_LISTE"+R(
	const uxx ib = 2ul*(uxx)(fbi>>5);
	const uint maske = f_maske[ib];
	const uint l = (uint)(fbi&31);
	if(((maske>>l)&1u)==0u) return false; // keine Wandsolidzelle: kein Slot, F ist dort konstruktiv 0
	*slot = (uxx)(f_maske[ib+1ul] + popcount(maske & ((1u<<l)-1u)));
)+"#else"+R(
	*slot = fbi; // Vollfeld-Arm: Slot = BBox-Index, wortgleich zum Stand vor dem 03.09.
)+"#endif"+R( // F_LISTE
	return true;
}
float3 load3_F(const global float* F, const global uint* f_maske, const uxx n) {
	uxx s; if(!f_slot(f_maske, n, &s)) return (float3)(0.0f, 0.0f, 0.0f);
	const ulong st = F_STRIDE;
	return (float3)(F[s], F[st+(ulong)s], F[2ul*st+(ulong)s]);
}
void store3_F(global float* F, const global uint* f_maske, const uxx n, const float3 v, global uint* hits) {
	uxx s;
	if(!f_slot(f_maske, n, &s)) {
)+"#ifdef F_LISTE"+R(
		// Nur echt, wenn die Zelle IN der Box liegt: ausserhalb ist das Verwerfen der alte, richtige Pfad.
		uxx fbi_; if(f_bbox(n, &fbi_) && hits[77]<0xF0000000u) atomic_inc(&hits[77]); // Wirkpfad-Waechter: Soll 0
)+"#endif"+R( // F_LISTE
		return;
	}
	const ulong st = F_STRIDE;
	F[s]=v.x; F[st+(ulong)s]=v.y; F[2ul*st+(ulong)s]=v.z;
}

// FORK 03.09.2026 -- fac_idx ist eine BITMASKE MIT PRAEFIXSUMME, kein volles uint-Feld mehr.
// Vorher: ein uint je F-BBox-Zelle (4 mm: 160.106.544 Zellen = 610,8 MiB) fuer eine Belegung von
// 1,95 %. Jetzt: je 32 F-BBox-Zellen ein Paar -- [2b] = Maske (Bit i gesetzt = Zelle 32b+i traegt
// eine AKTIVE Facette), [2b+1] = Zahl der aktiven Facetten VOR Block b (exklusive Praefixsumme).
//   fid = base + popcount(maske & Bits unterhalb der eigenen Lane)
// Das ist ganzzahlig exakt und liefert GENAU dieselbe Nummerierung wie der alte Zaehler, solange
// fbi streng monoton mit der Facettenreihenfolge waechst. Der Host prueft das hart in
// alloc_facetten_domain (Waechter, kein Kommentar) -- damit ist der Umbau bit-identisch abnehmbar.
// KEIN Geschwindigkeitshebel: die eingesparten ~640 MB/Schritt sind gegen den DDF-Verkehr
// desselben Kernels (>=37,6 GB/Schritt) unter 2 %. Der Posten traegt sich ueber 573 MiB VRAM.
uint band_fid(const global uint* band_idx, const uxx fbi) { // ★ 08.09. SGS-BAND: dieselbe Bitmasken-Nummerierung wie fac_fid, eigener Puffer, IMMER Bitmaske (kein VOLL-Rueckschalter)
	const uxx ib = 2ul*(uxx)(fbi>>5);
	const uint l = (uint)(fbi&31);
	const uint maske = band_idx[ib];
	if(((maske>>l)&1u)==0u) return 0xFFFFFFFFu; // keine Bandzelle
	return band_idx[ib+1ul] + (uint)popcount(maske & ((1u<<l)-1u));
}
uint fac_fid(const global uint* fac_idx, const uxx fbi) {
)+"#ifdef FAC_IDX_VOLL"+R(
	return fac_idx[fbi]; // ★ Rueckschalter CFD_FAC_IDX_VOLL: die alte Vollfeldform, ein uint je Zelle
)+"#else"+R(
	const uxx ib = 2ul*(uxx)(fbi>>5);
	const uint l = (uint)(fbi&31);
	const uint maske = fac_idx[ib];
	if(((maske>>l)&1u)==0u) return 0xFFFFFFFFu; // keine oder markierte Facette
	return fac_idx[ib+1ul] + (uint)popcount(maske & ((1u<<l)-1u));
)+"#endif"+R( // FAC_IDX_VOLL
}
)+"#endif"+R( // FORCE_FIELD
)+"#ifdef SPARSE_TILES"+R(
// FORK -- Block-Tiling (sparse solid). fi wird nur fuer AKTIVE Tiles alloziert; eine Tile gilt als tot,
// wenn sie SAMT 2-Zell-Halo vollstaendig solid ist. Layout tile-major SoA:
//     fi[ slot*(T^3*Q) + i*T^3 + local ],  slot = tile_slot[tile_id],  tot = 0xFFFFFFFF
// Invariante: eine Fluidzelle hat NIE einen Nachbarn in einer toten Tile -- sonst waere deren Halo nicht
// voll solid. Der 2-Halo (nicht 1) ist noetig, weil update_force_field via load_f die Nachbarn
// wand-adjazenter SOLID-Zellen liest, also bis zu 2 Zellen weit.
//
// Der Trade ist vermessen und strukturell, nicht wegoptimierbar (zwei Anlaeufe brachten zusammen +5 %):
// jeder Nachbarzugriff braucht einen zusaetzlichen tile_slot-Gather, und der Registerdruck senkt die
// Occupancy. Der groessere Teil der Strafe ist aber die DDF-Kontiguitaet: beim flachen Dispatch decken
// 64 Threads bei T=8 acht VERSCHIEDENE Tiles ab, also 8 mal 16 B statt 128 B am Stueck.
//     ERSPARNIS in v2, gemessen 11.09.2026 am Flag-Export des 4-mm-Laufs:
//     T=8 : 1287,8 MiB frei      T=16 : 450,1 MiB frei      T=64 und T=128 KOSTEN Speicher
//     (Aufrundungspolster 3,8 bzw. 6,7 GiB). Obergrenze jeder Halo-2-Kachelung: 2035,4 MiB.
//     DURCHSATZ: die frueher hier genannten -40 % / -28 % sind V1-Zahlen und in v2 NIE gemessen.
//     V1s Workgroup=Tile-Dispatch, der sie auf -12 % / -9 % senkte, ist hier NICHT portiert; er
//     spillt in v2 1152 B, weil sein geteiltes cbj-Array die Rang-1-Remat aufhebt. Einzelheiten in
//     PERFORMANCE-ANALYSE-2026-09-11.md Teil 3.4 und 3.5.
// Physikalisch bit-neutral (an der Kugel verifiziert). Default AUS -- es ist ein VRAM-gegen-Tempo-Regler
// fuer Faelle, die sonst nicht in den Speicher passen, kein genereller Gewinn. Das Nahfeld hat KEIN
// Verlangsamungsbudget: es traegt 95,8 % des Grobschritts, jede Verlangsamung schlaegt sofort durch.
ulong index_f_impl(const uxx n, const uint i, const global uint* tile_slot) {
	const uint x = (uint)((ulong)n % (ulong)def_Nx);
	const uint y = (uint)(((ulong)n / (ulong)def_Nx) % (ulong)def_Ny);
	const uint z = (uint)((ulong)n / ((ulong)def_Nx*(ulong)def_Ny));
	const uint tx = x/def_TILE, ty = y/def_TILE, tz = z/def_TILE;
	const uint slot = tile_slot[tx + def_TILES_X*(ty + def_TILES_Y*tz)];
	if(slot==def_TILE_DEAD) return 0ul; // tote Tile -> Papierkorb-Slot 0 (echte Slots zaehlen ab 1)
	const uint loc = (x - tx*def_TILE) + def_TILE*((y - ty*def_TILE) + def_TILE*(z - tz*def_TILE));
	return (ulong)slot*((ulong)def_TILE*def_TILE*def_TILE*def_velocity_set) + (ulong)i*((ulong)def_TILE*def_TILE*def_TILE) + (ulong)loc;
}
bool is_dead_tile(const uxx n, const global uint* tile_slot) { // true -> Tile gedroppt, fi-Zugriff verboten
	const uint x = (uint)((ulong)n % (ulong)def_Nx);
	const uint y = (uint)(((ulong)n / (ulong)def_Nx) % (ulong)def_Ny);
	const uint z = (uint)((ulong)n / ((ulong)def_Nx*(ulong)def_Ny));
	return tile_slot[(x/def_TILE) + def_TILES_X*((y/def_TILE) + def_TILES_Y*(z/def_TILE))]==def_TILE_DEAD;
}
// Richtungsunabhaengiger Teil des Index, damit load_f/store_f ihn 1x statt 10x pro Zelle rechnen.
ulong cell_base(const uxx n, const global uint* tile_slot) {
	const uint x = (uint)((ulong)n % (ulong)def_Nx);
	const uint y = (uint)(((ulong)n / (ulong)def_Nx) % (ulong)def_Ny);
	const uint z = (uint)((ulong)n / ((ulong)def_Nx*(ulong)def_Ny));
	const uint tx = x/def_TILE, ty = y/def_TILE, tz = z/def_TILE;
	const uint slot = tile_slot[tx + def_TILES_X*(ty + def_TILES_Y*tz)];
	if(slot==def_TILE_DEAD) return 0ul; // tote Tile -> Papierkorb-Slot 0; Schreibzugriffe an Nachbarindizes
	                                     // in toten Tiles landen so harmlos statt in einer echten Tile
	const uint loc = (x - tx*def_TILE) + def_TILE*((y - ty*def_TILE) + def_TILE*(z - tz*def_TILE));
	return (ulong)slot*((ulong)def_TILE*def_TILE*def_TILE*def_velocity_set) + (ulong)loc;
}
)+"#else"+R(
)+R(ulong index_f(const uxx n, const uint i) { // 64-bit indexing for DDFs
	return (ulong)i*def_N+(ulong)n; // SoA (>2x faster on GPUs)
}
)+"#endif"+R( // SPARSE_TILES
)+R(float c(const uint i) { // avoid constant keyword by encapsulating data in function which gets inlined by compiler
	const float c[3u*def_velocity_set] = {
)+"#if defined(D2Q9)"+R(
		0, 1,-1, 0, 0, 1,-1, 1,-1, // x
		0, 0, 0, 1,-1, 1,-1,-1, 1, // y
		0, 0, 0, 0, 0, 0, 0, 0, 0  // z
)+"#elif defined(D3Q15)"+R(
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1,-1, 1, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1  // z
)+"#elif defined(D3Q19)"+R(
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1  // z
)+"#elif defined(D3Q27)"+R(
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1  // z
)+"#endif"+R( // D3Q27
	};
	return c[i];
}
)+R(float w(const uint i) { // avoid constant keyword by encapsulating data in function which gets inlined by compiler
	const float w[def_velocity_set] = { def_w0, // velocity set weights
)+"#if defined(D2Q9)"+R(
		def_ws, def_ws, def_ws, def_ws, def_we, def_we, def_we, def_we
)+"#elif defined(D3Q15)"+R(
		def_ws, def_ws, def_ws, def_ws, def_ws, def_ws,
		def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc
)+"#elif defined(D3Q19)"+R(
		def_ws, def_ws, def_ws, def_ws, def_ws, def_ws,
		def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we
)+"#elif defined(D3Q27)"+R(
		def_ws, def_ws, def_ws, def_ws, def_ws, def_ws,
		def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we, def_we,
		def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc, def_wc
)+"#endif"+R( // D3Q27
	};
	return w[i];
}
)+R(void calculate_indices(const uxx n, uxx* x0, uxx* xp, uxx* xm, uxx* y0, uxx* yp, uxx* ym, uxx* z0, uxx* zp, uxx* zm) {
	const uint3 xyz = coordinates(n);
	*x0 = (uxx)   xyz.x; // pre-calculate indices (periodic boundary conditions)
	*xp = (uxx) ((xyz.x       +1u)%def_Nx);
	*xm = (uxx) ((xyz.x+def_Nx-1u)%def_Nx);
	*y0 = (uxx)(  xyz.y                   *def_Nx);
	*yp = (uxx)(((xyz.y       +1u)%def_Ny)*def_Nx);
	*ym = (uxx)(((xyz.y+def_Ny-1u)%def_Ny)*def_Nx);
	*z0 = (uxx)   xyz.z                   *(uxx)(def_Ny*def_Nx);
	*zp = (uxx) ((xyz.z       +1u)%def_Nz)*(uxx)(def_Ny*def_Nx);
	*zm = (uxx) ((xyz.z+def_Nz-1u)%def_Nz)*(uxx)(def_Ny*def_Nx);
} // calculate_indices()
)+R(void neighbors(const uxx n, uxx* j) { // calculate neighbor indices
	uxx x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	j[0] = n;
)+"#if defined(D2Q9)"+R(
	j[ 1] = xp+y0; j[ 2] = xm+y0; // +00 -00
	j[ 3] = x0+yp; j[ 4] = x0+ym; // 0+0 0-0
	j[ 5] = xp+yp; j[ 6] = xm+ym; // ++0 --0
	j[ 7] = xp+ym; j[ 8] = xm+yp; // +-0 -+0
)+"#elif defined(D3Q15)"+R(
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+zp; j[ 8] = xm+ym+zm; // +++ ---
	j[ 9] = xp+yp+zm; j[10] = xm+ym+zp; // ++- --+
	j[11] = xp+ym+zp; j[12] = xm+yp+zm; // +-+ -+-
	j[13] = xm+yp+zp; j[14] = xp+ym+zm; // -++ +--
)+"#elif defined(D3Q19)"+R(
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+z0; j[ 8] = xm+ym+z0; // ++0 --0
	j[ 9] = xp+y0+zp; j[10] = xm+y0+zm; // +0+ -0-
	j[11] = x0+yp+zp; j[12] = x0+ym+zm; // 0++ 0--
	j[13] = xp+ym+z0; j[14] = xm+yp+z0; // +-0 -+0
	j[15] = xp+y0+zm; j[16] = xm+y0+zp; // +0- -0+
	j[17] = x0+yp+zm; j[18] = x0+ym+zp; // 0+- 0-+
)+"#elif defined(D3Q27)"+R(
	j[ 1] = xp+y0+z0; j[ 2] = xm+y0+z0; // +00 -00
	j[ 3] = x0+yp+z0; j[ 4] = x0+ym+z0; // 0+0 0-0
	j[ 5] = x0+y0+zp; j[ 6] = x0+y0+zm; // 00+ 00-
	j[ 7] = xp+yp+z0; j[ 8] = xm+ym+z0; // ++0 --0
	j[ 9] = xp+y0+zp; j[10] = xm+y0+zm; // +0+ -0-
	j[11] = x0+yp+zp; j[12] = x0+ym+zm; // 0++ 0--
	j[13] = xp+ym+z0; j[14] = xm+yp+z0; // +-0 -+0
	j[15] = xp+y0+zm; j[16] = xm+y0+zp; // +0- -0+
	j[17] = x0+yp+zm; j[18] = x0+ym+zp; // 0+- 0-+
	j[19] = xp+yp+zp; j[20] = xm+ym+zm; // +++ ---
	j[21] = xp+yp+zm; j[22] = xm+ym+zp; // ++- --+
	j[23] = xp+ym+zp; j[24] = xm+yp+zm; // +-+ -+-
	j[25] = xm+yp+zp; j[26] = xp+ym+zm; // -++ +--
)+"#endif"+R( // D3Q27
} // neighbors()

)+R(float3 load3(const global float* p, const uxx n) {
	return (float3)(p[n], p[def_N+(ulong)n], p[2ul*def_N+(ulong)n]);
}
)+R(void store3(global float* p, const uxx n, const float3 v) { // ★ 12.09.: seit store3_u OHNE Aufrufstelle (load3 traegt weiter F und GRAPHICS). Bleibt fuer den Fall, dass GRAPHICS wiederbelebt wird.
	p[                 n] = v.x;
	p[    def_N+(ulong)n] = v.y;
	p[2ul*def_N+(ulong)n] = v.z;
}

)+R(float3 load3_u(const global velxx* p, const uxx n) {
	return (float3)(load_u(p, n), load_u(p, def_N+(ulong)n), load_u(p, 2ul*def_N+(ulong)n));
}
)+R(void store3_u(global velxx* p, const uxx n, const float3 v) {
	store_u(p,                  n, v.x);
	store_u(p,     def_N+(ulong)n, v.y);
	store_u(p, 2ul*def_N+(ulong)n, v.z);
}
)+R(float3 closest_u(const global float* u, const float3 p) { // return velocity of closest lattice point to point p
	return load3(u, index(closest_coordinates(p)));
}
)+R(float3 interpolate_u(const global float* u, const float3 p) { // trilinear interpolation of velocity at point p
	const float xa=p.x-0.5f+1.5f*(float)def_Nx, ya=p.y-0.5f+1.5f*(float)def_Ny, za=p.z-0.5f+1.5f*(float)def_Nz; // subtract lattice offsets
	const uint xb=(uint)xa, yb=(uint)ya, zb=(uint)za; // integer casting to find bottom left corner
	const float3 pn = (float3)(xa-(float)xb, ya-(float)yb, za-(float)zb); // calculate interpolation factors
	float3 un[8]; // velocitiy at unit cube corner points
	for(uint c=0u; c<8u; c++) { // count over eight corner points
		const uint i=(c&0x04u)>>2, j=(c&0x02u)>>1, k=c&0x01u; // disassemble c into corner indices ijk
		const uint x=(xb+i)%def_Nx, y=(yb+j)%def_Ny, z=(zb+k)%def_Nz; // calculate corner lattice positions
		const uxx n = (uxx)x+(uxx)(y+z*def_Ny)*(uxx)def_Nx; // calculate lattice linear index
		un[c] = load3(u, n); // load velocity from lattice point
	}
	return trilinear3(pn, un); // perform trilinear interpolation
} // interpolate_u()
)+R(float calculate_Q_cached(const float3 u0, const float3 u1, const float3 u2, const float3 u3, const float3 u4, const float3 u5) { // Q-criterion
	const float s_xx2=u0.x-u1.x, duydx=u0.y-u1.y, duzdx=u0.z-u1.z; // du/dx = (u2-u0)/2
	const float duxdy=u2.x-u3.x, s_yy2=u2.y-u3.y, duzdy=u2.z-u3.z; // s_xx2 = s_xx/2, s_yy2 = s_yy/2, s_zz2 = s_zz/2
	const float duxdz=u4.x-u5.x, duydz=u4.y-u5.y, s_zz2=u4.z-u5.z;
	const float omega_xy=duxdy-duydx, omega_xz=duxdz-duzdx, omega_yz=duydz-duzdy; // antisymmetric tensor, omega_xx = omega_yy = omega_zz = 0
	const float s_xy=duxdy+duydx, s_xz=duxdz+duzdx, s_yz=duydz+duzdy; // symmetric tensor
	const float omega2 = fma(omega_xy, omega_xy, fma(omega_xz, omega_xz, sq(omega_yz))); // ||omega||_2^2
	const float s2 = 2.0f*fma(s_xx2, s_xx2, fma(s_yy2, s_yy2, sq(s_zz2)))+fma(s_xy, s_xy, fma(s_xz, s_xz, sq(s_yz))); // ||s||_2^2
	return 0.25f*(omega2-s2); // Q = 1/2*(||omega||_2^2-||s||_2^2), addidional factor 1/2 from cental finite differences of velocity
} // calculate_Q_cached()
)+R(float calculate_Q(const uxx n, const global float* u) { // Q-criterion
	uxx x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	uxx j[6];
	j[0] = xp+y0+z0; j[1] = xm+y0+z0; // +00 -00
	j[2] = x0+yp+z0; j[3] = x0+ym+z0; // 0+0 0-0
	j[4] = x0+y0+zp; j[5] = x0+y0+zm; // 00+ 00-
	return calculate_Q_cached(load3(u, j[0]), load3(u, j[1]), load3(u, j[2]), load3(u, j[3]), load3(u, j[4]), load3(u, j[5]));
} // calculate_Q()

)+R(void calculate_f_eq(const float rho, float ux, float uy, float uz, float* feq) { // calculate f_equilibrium from density and velocity field (perturbation method / DDF-shifting)
	const float rhom1 = rho-1.0f; // rhom1 is arithmetic optimization to minimize digit extinction
)+"#ifndef D2Q9"+R( // 3D
	const float c3 = -3.0f*(sq(ux)+sq(uy)+sq(uz)); // c3 = -2*sq(u)/(2*sq(c))
	uz *= 3.0f; // only needed for 3D
)+"#else"+R( // D2Q9
	const float c3 = -3.0f*(sq(ux)+sq(uy)); // c3 = -2*sq(u)/(2*sq(c))
)+"#endif"+R( // D2Q9
	ux *= 3.0f;
	uy *= 3.0f;
	feq[ 0] = def_w0*fma(rho, 0.5f*c3, rhom1); // 000 (identical for all velocity sets)
)+"#if defined(D2Q9)"+R(
	const float u0=ux+uy, u1=ux-uy; // these pre-calculations make manual unrolling require less FLOPs
	const float rhos=def_ws*rho, rhoe=def_we*rho, rhom1s=def_ws*rhom1, rhom1e=def_we*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 6] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +-0 -+0
)+"#elif defined(D3Q15)"+R(
	const float u0=ux+uy+uz, u1=ux+uy-uz, u2=ux-uy+uz, u3=-ux+uy+uz;
	const float rhos=def_ws*rho, rhoc=def_wc*rho, rhom1s=def_ws*rhom1, rhom1c=def_wc*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[ 7] = fma(rhoc, fma(0.5f, fma(u0, u0, c3), u0), rhom1c); feq[ 8] = fma(rhoc, fma(0.5f, fma(u0, u0, c3), -u0), rhom1c); // +++ ---
	feq[ 9] = fma(rhoc, fma(0.5f, fma(u1, u1, c3), u1), rhom1c); feq[10] = fma(rhoc, fma(0.5f, fma(u1, u1, c3), -u1), rhom1c); // ++- --+
	feq[11] = fma(rhoc, fma(0.5f, fma(u2, u2, c3), u2), rhom1c); feq[12] = fma(rhoc, fma(0.5f, fma(u2, u2, c3), -u2), rhom1c); // +-+ -+-
	feq[13] = fma(rhoc, fma(0.5f, fma(u3, u3, c3), u3), rhom1c); feq[14] = fma(rhoc, fma(0.5f, fma(u3, u3, c3), -u3), rhom1c); // -++ +--
)+"#elif defined(D3Q19)"+R(
	const float u0=ux+uy, u1=ux+uz, u2=uy+uz, u3=ux-uy, u4=ux-uz, u5=uy-uz;
	const float rhos=def_ws*rho, rhoe=def_we*rho, rhom1s=def_ws*rhom1, rhom1e=def_we*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
	feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
	feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
	feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
	feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
)+"#elif defined(D3Q27)"+R(
	const float u0=ux+uy, u1=ux+uz, u2=uy+uz, u3=ux-uy, u4=ux-uz, u5=uy-uz, u6=ux+uy+uz, u7=ux+uy-uz, u8=ux-uy+uz, u9=-ux+uy+uz;
	const float rhos=def_ws*rho, rhoe=def_we*rho, rhoc=def_wc*rho, rhom1s=def_ws*rhom1, rhom1e=def_we*rhom1, rhom1c=def_wc*rhom1;
	feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
	feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
	feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
	feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
	feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
	feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
	feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
	feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
	feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
	feq[19] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), u6), rhom1c); feq[20] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), -u6), rhom1c); // +++ ---
	feq[21] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), u7), rhom1c); feq[22] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), -u7), rhom1c); // ++- --+
	feq[23] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), u8), rhom1c); feq[24] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), -u8), rhom1c); // +-+ -+-
	feq[25] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), u9), rhom1c); feq[26] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), -u9), rhom1c); // -++ +--
)+"#endif"+R( // D3Q27
} // calculate_f_eq()

)+R(void calculate_rho_u(const float* f, float* rhon, float* uxn, float* uyn, float* uzn) { // calculate density and velocity fields from fi
	float rho=f[0], ux, uy, uz;
	for(uint i=1u; i<def_velocity_set; i++) rho += f[i]; // calculate density from fi
	rho += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up fi (perturbation method / DDF-shifting)
)+"#if defined(D2Q9)"+R(
	ux = f[1]-f[2]+f[5]-f[6]+f[7]-f[8]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[3]-f[4]+f[5]-f[6]+f[8]-f[7];
	uz = 0.0f;
)+"#elif defined(D3Q15)"+R(
	ux = f[ 1]-f[ 2]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[11]-f[12]+f[14]-f[13]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[ 3]-f[ 4]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[12]-f[11]+f[13]-f[14];
	uz = f[ 5]-f[ 6]+f[ 7]-f[ 8]+f[10]-f[ 9]+f[11]-f[12]+f[13]-f[14];
)+"#elif defined(D3Q19)"+R(
	ux = f[ 1]-f[ 2]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[13]-f[14]+f[15]-f[16]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[ 3]-f[ 4]+f[ 7]-f[ 8]+f[11]-f[12]+f[14]-f[13]+f[17]-f[18];
	uz = f[ 5]-f[ 6]+f[ 9]-f[10]+f[11]-f[12]+f[16]-f[15]+f[18]-f[17];
)+"#elif defined(D3Q27)"+R(
	ux = f[ 1]-f[ 2]+f[ 7]-f[ 8]+f[ 9]-f[10]+f[13]-f[14]+f[15]-f[16]+f[19]-f[20]+f[21]-f[22]+f[23]-f[24]+f[26]-f[25]; // calculate velocity from fi (alternating + and - for best accuracy)
	uy = f[ 3]-f[ 4]+f[ 7]-f[ 8]+f[11]-f[12]+f[14]-f[13]+f[17]-f[18]+f[19]-f[20]+f[21]-f[22]+f[24]-f[23]+f[25]-f[26];
	uz = f[ 5]-f[ 6]+f[ 9]-f[10]+f[11]-f[12]+f[16]-f[15]+f[18]-f[17]+f[19]-f[20]+f[22]-f[21]+f[23]-f[24]+f[25]-f[26];
)+"#endif"+R( // D3Q27
)+"#ifdef RHO_CLAMP"+R(
	// ★★ DICHTE-KLEMME, aus V1 nachgezogen 2026-08-09. Sie steht HIER, vor der Division -- das ist
	// der ganze Punkt: u = j/rho explodiert, sobald rho gegen null oder negativ laeuft. Geklemmt wird
	// rho, geteilt wird durch das GEKLEMMTE rho; der Impuls j = rho*u bleibt damit erhalten.
	// V1 hat sie am 2026-06-25 gegen den Druckdipol ueber dem bewegten Boden eingefuehrt und
	// dokumentiert sie als DEN Fix dafuer. V2 hatte sie nicht -- und der Doppel-Domaenen-Lauf ist
	// am 2026-08-09 bei 0,003 s mit NaN gestorben, also an genau diesem Mechanismus.
	// Die Grenzen sind physikalisch universell, kein Fallknopf: bei kleiner Machzahl ist rho = 1 +- 0,02,
	// die Klemme greift also NUR bei grober numerischer Instabilitaet. Ein gesundes Feld bleibt
	// bit-identisch -- genau das macht sie als Waechter brauchbar.
	rho = fmin(RHO_CLAMP_MAX, fmax(RHO_CLAMP_MIN, rho));
)+"#endif"+R( // RHO_CLAMP
	*rhon = rho;
	*uxn = ux/rho;
	*uyn = uy/rho;
	*uzn = uz/rho;
} // calculate_rho_u()

)+"#ifdef VOLUME_FORCE"+R(
)+R(void calculate_forcing_terms(const float ux, const float uy, const float uz, const float fx, const float fy, const float fz, float* Fin) { // calculate volume force terms Fin from velocity field (Guo forcing, Krueger p.233f)
)+"#ifdef D2Q9"+R(
	const float uF = -0.33333334f*fma(ux, fx, uy*fy); // 2D
)+"#else"+R( // D2Q9
	const float uF = -0.33333334f*fma(ux, fx, fma(uy, fy, uz*fz)); // 3D
)+"#endif"+R( // D2Q9
	Fin[0] = 9.0f*def_w0*uF ; // 000 (identical for all velocity sets)
	for(uint i=1u; i<def_velocity_set; i++) { // loop is entirely unrolled by compiler, no unnecessary FLOPs are happening
		Fin[i] = 9.0f*w(i)*fma(c(i)*fx+c(def_velocity_set+i)*fy+c(2u*def_velocity_set+i)*fz, c(i)*ux+c(def_velocity_set+i)*uy+c(2u*def_velocity_set+i)*uz+0.33333334f, uF);
	}
} // calculate_forcing_terms()
)+"#endif"+R( // VOLUME_FORCE

)+"#ifdef MOVING_BOUNDARIES"+R(
)+R(void apply_moving_boundaries(float* fhn, const uxx* j, const global velxx* u, const global uchar* flags) { // apply Dirichlet velocity boundaries if necessary (Krueger p.180, rho_solid=1)
	uxx ji; // reads velocities of only neighboring boundary cells, which do not change during simulation
	for(uint i=1u; i<def_velocity_set; i+=2u) { // loop is entirely unrolled by compiler, no unnecessary memory access is happening
		const float w6 = -6.0f*w(i); // w6 = -2*w_i*rho_wall/c^2, w(i) = w(i+1) if i is odd, rho_wall is assumed as rho_avg=1 (necessary choice to assure mass conservation)
		ji = j[i+1u]; fhn[i   ] = (flags[ji]&TYPE_BO)==TYPE_S ? fma(w6, c(i+1u)*load_u(u, ji)+c(def_velocity_set+i+1u)*load_u(u, def_N+(ulong)ji)+c(2u*def_velocity_set+i+1u)*load_u(u, 2ul*def_N+(ulong)ji), fhn[i   ]) : fhn[i   ]; // boundary : regular
		ji = j[i   ]; fhn[i+1u] = (flags[ji]&TYPE_BO)==TYPE_S ? fma(w6, c(i   )*load_u(u, ji)+c(def_velocity_set+i   )*load_u(u, def_N+(ulong)ji)+c(2u*def_velocity_set+i   )*load_u(u, 2ul*def_N+(ulong)ji), fhn[i+1u]) : fhn[i+1u];
	}
} // apply_moving_boundaries()
)+"#endif"+R( // MOVING_BOUNDARIES

)+"#ifdef SURFACE"+R(
)+R(void average_neighbors_non_gas(const uxx n, const global float* rho, const global float* u, const global uchar* flags, float* rhon, float* uxn, float* uyn, float* uzn) { // calculate average density and velocity of neighbors of cell n
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	float rhot=0.0f, uxt=0.0f, uyt=0.0f, uzt=0.0f, counter=0.0f; // average over all fluid/interface neighbors
	for(uint i=1u; i<def_velocity_set; i++) {
		const uchar flagsji_sus = flags[j[i]]&(TYPE_SU|TYPE_S); // extract SURFACE flags
		if(flagsji_sus==TYPE_F||flagsji_sus==TYPE_I||flagsji_sus==TYPE_IF) { // fluid or interface or (interface->fluid) neighbor
			counter += 1.0f;
			rhot += rho[               j[i]];
			uxt  += u[                 j[i]];
			uyt  += u[    def_N+(ulong)j[i]];
			uzt  += u[2ul*def_N+(ulong)j[i]];
		}
	}
	*rhon = counter>0.0f ? rhot/counter : 1.0f;
	*uxn  = counter>0.0f ? uxt /counter : 0.0f;
	*uyn  = counter>0.0f ? uyt /counter : 0.0f;
	*uzn  = counter>0.0f ? uzt /counter : 0.0f;
}
)+R(void average_neighbors_fluid(const uxx n, const global float* rho, const global float* u, const global uchar* flags, float* rhon, float* uxn, float* uyn, float* uzn) { // calculate average density and velocity of neighbors of cell n
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	float rhot=0.0f, uxt=0.0f, uyt=0.0f, uzt=0.0f, counter=0.0f; // average over all fluid/interface neighbors
	for(uint i=1u; i<def_velocity_set; i++) {
		const uchar flagsji_su = flags[j[i]]&TYPE_SU;
		if(flagsji_su==TYPE_F) { // fluid neighbor
			counter += 1.0f;
			rhot += rho[               j[i]];
			uxt  += u[                 j[i]];
			uyt  += u[    def_N+(ulong)j[i]];
			uzt  += u[2ul*def_N+(ulong)j[i]];
		}
	}
	*rhon = counter>0.0f ? rhot/counter : 1.0f;
	*uxn  = counter>0.0f ? uxt /counter : 0.0f;
	*uyn  = counter>0.0f ? uyt /counter : 0.0f;
	*uzn  = counter>0.0f ? uzt /counter : 0.0f;
}
)+R(float calculate_phi(const float rhon, const float massn, const uchar flagsn) { // calculate fill level
	return flagsn&TYPE_F ? 1.0f : flagsn&TYPE_I ? rhon>0.0f ? clamp(massn/rhon, 0.0f, 1.0f) : 0.5f : 0.0f;
}
)+R(float3 calculate_normal_py(const float* phij) { // calculate surface normal vector (Parker-youngs approximation, more accurate, works only for D3Q27 neighborhood)
	float3 n; // normal vector
)+"#ifdef D2Q9"+R(
	n.x = 2.0f*(phij[2]-phij[1])+phij[6]-phij[5]+phij[8]-phij[7];
	n.y = 2.0f*(phij[4]-phij[3])+phij[6]-phij[5]+phij[7]-phij[8];
	n.z = 0.0f;
)+"#else"+R( // D2Q9
	n.x = 4.0f*(phij[ 2]-phij[ 1])+2.0f*(phij[ 8]-phij[ 7]+phij[10]-phij[ 9]+phij[14]-phij[13]+phij[16]-phij[15])+phij[20]-phij[19]+phij[22]-phij[21]+phij[24]-phij[23]+phij[25]-phij[26];
	n.y = 4.0f*(phij[ 4]-phij[ 3])+2.0f*(phij[ 8]-phij[ 7]+phij[12]-phij[11]+phij[13]-phij[14]+phij[18]-phij[17])+phij[20]-phij[19]+phij[22]-phij[21]+phij[23]-phij[24]+phij[26]-phij[25];
	n.z = 4.0f*(phij[ 6]-phij[ 5])+2.0f*(phij[10]-phij[ 9]+phij[12]-phij[11]+phij[15]-phij[16]+phij[17]-phij[18])+phij[20]-phij[19]+phij[21]-phij[22]+phij[24]-phij[23]+phij[26]-phij[25];
)+"#endif"+R( // D2Q9
	return normalize(n);
}
)+R(float plic_cube_reduced(const float V, const float n1, const float n2, const float n3) { // optimized solution from SZ and Kawano, source: https://doi.org/10.3390/computation10020021
	const float n12=n1+n2, n3V=n3*V;
	if(n12<=2.0f*n3V) return n3V+0.5f*n12; // case (5)
	const float sqn1=sq(n1), n26=6.0f*n2, v1=sqn1/n26; // after case (5) check n2>0 is true
	if(v1<=n3V && n3V<v1+0.5f*(n2-n1)) return 0.5f*(n1+sqrt(sqn1+8.0f*n2*(n3V-v1))); // case (2)
	const float V6 = n1*n26*n3V;
	if(n3V<v1) return cbrt(V6); // case (1)
	const float v3 = n3<n12 ? (sq(n3)*(3.0f*n12-n3)+sqn1*(n1-3.0f*n3)+sq(n2)*(n2-3.0f*n3))/(n1*n26) : 0.5f*n12; // after case (2) check n1>0 is true
	const float sqn12=sqn1+sq(n2), V6cbn12=V6-cb(n1)-cb(n2);
	const bool case34 = n3V<v3; // true: case (3), false: case (4)
	const float a = case34 ? V6cbn12 : 0.5f*(V6cbn12-cb(n3));
	const float b = case34 ?   sqn12 : 0.5f*(sqn12+sq(n3));
	const float c = case34 ?     n12 : 0.5f;
	const float t = sqrt(sq(c)-b);
	return c-2.0f*t*sin(0.33333334f*asin((cb(c)-0.5f*a-1.5f*b*c)/cb(t)));
}
)+R(float plic_cube(const float V0, const float3 n) { // unit cube - plane intersection: volume V0 in [0,1], normal vector n -> plane offset d0
	const float ax=fabs(n.x), ay=fabs(n.y), az=fabs(n.z), V=0.5f-fabs(V0-0.5f), l=ax+ay+az; // eliminate symmetry cases, normalize n using L1 norm
	const float n1 = fmin(fmin(ax, ay), az)/l;
	const float n3 = fmax(fmax(ax, ay), az)/l;
	const float n2 = fdim(1.0f, n1+n3); // ensure n2>=0
	const float d = plic_cube_reduced(V, n1, n2, n3); // calculate PLIC with reduced symmetry
	return l*copysign(0.5f-d, V0-0.5f); // rescale result and apply symmetry for V0>0.5
}
)+R(void get_remaining_neighbor_phij(const uxx n, const float* phit, const global float* phi, float* phij) { // get remaining phij for D3Q27 neighborhood
)+"#ifndef D3Q27"+R(
	uxx x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
)+"#endif"+R( // D3Q27
)+"#if defined(D3Q15)"+R(
	uxx j[12]; // calculate neighbor indices
	j[ 0] = xp+yp+z0; j[ 1] = xm+ym+z0; // ++0 --0
	j[ 2] = xp+y0+zp; j[ 3] = xm+y0+zm; // +0+ -0-
	j[ 4] = x0+yp+zp; j[ 5] = x0+ym+zm; // 0++ 0--
	j[ 6] = xp+ym+z0; j[ 7] = xm+yp+z0; // +-0 -+0
	j[ 8] = xp+y0+zm; j[ 9] = xm+y0+zp; // +0- -0+
	j[10] = x0+yp+zm; j[11] = x0+ym+zp; // 0+- 0-+
	for(uint i= 0u; i< 7u; i++) phij[i] = phit[i];
	for(uint i= 7u; i<19u; i++) phij[i] = phi[j[i-7u]];
	for(uint i=19u; i<27u; i++) phij[i] = phit[i-12u];
)+"#elif defined(D3Q19)"+R(
	uxx j[8]; // calculate remaining neighbor indices
	j[0] = xp+yp+zp; j[1] = xm+ym+zm; // +++ ---
	j[2] = xp+yp+zm; j[3] = xm+ym+zp; // ++- --+
	j[4] = xp+ym+zp; j[5] = xm+yp+zm; // +-+ -+-
	j[6] = xm+yp+zp; j[7] = xp+ym+zm; // -++ +--
	for(uint i= 0u; i<19u; i++) phij[i] = phit[i];
	for(uint i=19u; i<27u; i++) phij[i] = phi[j[i-19u]];
)+"#elif defined(D3Q27)"+R(
	for(uint i=0u; i<def_velocity_set; i++) phij[i] = phit[i];
)+"#endif"+R( // D3Q27
}
)+R(float c_D3Q27(const uint i) { // avoid constant keyword by encapsulating data in function which gets inlined by compiler
	const float c[3*27] = {
		0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, // x
		0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, // y
		0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1  // z
	};
	return c[i];
}
)+R(float calculate_curvature(const uxx n, const float* phit, const global float* phi) { // calculate surface curvature, always use D3Q27 stencil here, source: https://doi.org/10.3390/computation10020021
)+"#ifndef D2Q9"+R(
	float phij[27];
	get_remaining_neighbor_phij(n, phit, phi, phij); // complete neighborhood from whatever velocity set is selected to D3Q27
	const float3 bz = calculate_normal_py(phij); // new coordinate system: bz is normal to surface, bx and by are tangent to surface
	const float3 rn = (float3)(0.56270900f, 0.32704452f, 0.75921047f); // random normalized vector that is just by random chance not collinear with bz
	const float3 by = normalize(cross(bz, rn)); // normalize() is necessary here because bz and rn are not perpendicular
	const float3 bx = cross(by, bz);
	uint number = 0; // number of neighboring interface points
	float3 p[24]; // number of neighboring interface points is less or equal than than 26 minus 1 gas and minus 1 fluid point = 24
	const float center_offset = plic_cube(phij[0], bz); // calculate z-offset PLIC of center point only once
	for(uint i=1u; i<27u; i++) { // iterate over neighbors, no loop unrolling here (50% better perfoemance without loop unrolling)
		if(phij[i]>0.0f&&phij[i]<1.0f) { // limit neighbors to interface cells
			const float3 ei = (float3)(c_D3Q27(i), c_D3Q27(27u+i), c_D3Q27(2u*27u+i)); // assume neighbor normal vector is the same as center normal vector
			const float offset = plic_cube(phij[i], bz)-center_offset;
			p[number++] = (float3)(dot(ei, bx), dot(ei, by), dot(ei, bz)+offset); // do coordinate system transformation into (x, y, f(x,y)) and apply PLIC pffsets
		}
	}
	float M[25], x[5]={0.0f,0.0f,0.0f,0.0f,0.0f}, b[5]={0.0f,0.0f,0.0f,0.0f,0.0f};
	for(uint i=0u; i<25u; i++) M[i] = 0.0f;
	for(uint i=0u; i<number; i++) { // f(x,y)=A*x2+B*y2+C*x*y+H*x+I*y, x=(A,B,C,H,I), Q=(x2,y2,x*y,x,y), M*x=b, M=Q*Q^T, b=Q*z
		const float x=p[i].x, y=p[i].y, z=p[i].z, x2=x*x, y2=y*y, x3=x2*x, y3=y2*y;
		/**/M[ 0]+=x2*x2; M[ 1]+=x2*y2; M[ 2]+=x3*y ; M[ 3]+=x3   ; M[ 4]+=x2*y ; b[0]+=x2   *z;
		/*M[ 5]+=x2*y2;*/ M[ 6]+=y2*y2; M[ 7]+=x *y3; M[ 8]+=x *y2; M[ 9]+=   y3; b[1]+=   y2*z;
		/*M[10]+=x3*y ; M[11]+=x *y3;*/ M[12]+=x2*y2; M[13]+=x2*y ; M[14]+=x *y2; b[2]+=x *y *z;
		/*M[15]+=x3   ; M[16]+=x *y2; M[17]+=x2*y ;*/ M[18]+=x2   ; M[19]+=x *y ; b[3]+=x    *z;
		/*M[20]+=x2*y ; M[21]+=   y3; M[22]+=x *y2; M[23]+=x *y ;*/ M[24]+=   y2; b[4]+=   y *z;
	}
	for(uint i=1u; i<5u; i++) { // use symmetry of matrix to save arithmetic operations
		for(uint j=0u; j<i; j++) M[i*5+j] = M[j*5+i];
	}
	if(number>=5u) lu_solve(M, x, b, 5, 5);
	else lu_solve(M, x, b, 5, min(5u, number)); // cannot do loop unrolling here -> slower -> extra if-else to avoid slowdown
	const float A=x[0], B=x[1], C=x[2], H=x[3], I=x[4];
	const float K = (A*(I*I+1.0f)+B*(H*H+1.0f)-C*H*I)*cb(rsqrt(H*H+I*I+1.0f)); // mean curvature of Monge patch (x, y, f(x, y))
)+"#else"+R( // D2Q9
	const float3 by = calculate_normal_py(phit); // new coordinate system: bz is normal to surface, bx and by are tangent to surface
	const float3 bx = cross(by, (float3)(0.0f, 0.0f, 1.0f)); // normalize() is necessary here because bz and rn are not perpendicular
	uint number = 0u; // number of neighboring interface points
	float2 p[6]; // number of neighboring interface points is less or equal than than 8 minus 1 gas and minus 1 fluid point = 6
	const float center_offset = plic_cube(phit[0], by); // calculate z-offset PLIC of center point only once
	for(uint i=1u; i<9u; i++) { // iterate over neighbors, no loop unrolling here (50% better perfoemance without loop unrolling)
		if(phit[i]>0.0f&&phit[i]<1.0f) { // limit neighbors to interface cells
			const float3 ei = (float3)(c(i), c(9u+i), 0.0f); // assume neighbor normal vector is the same as center normal vector
			const float offset = plic_cube(phit[i], by)-center_offset;
			p[number++] = (float2)(dot(ei, bx), dot(ei, by)+offset); // do coordinate system transformation into (x, f(x)) and apply PLIC pffsets
		}
	}
	float M[4]={0.0f,0.0f,0.0f,0.0f}, x[2]={0.0f,0.0f}, b[2]={0.0f,0.0f};
	for(uint i=0u; i<number; i++) { // f(x,y)=A*x2+H*x, x=(A,H), Q=(x2,x), M*x=b, M=Q*Q^T, b=Q*z
		const float x=p[i].x, y=p[i].y, x2=x*x, x3=x2*x;
		/**/M[0]+=x2*x2; M[1]+=x3; b[0]+=x2*y;
		/*M[2]+=x3   ;*/ M[3]+=x2; b[1]+=x *y;
	}
	M[2] = M[1]; // use symmetry of matrix to save arithmetic operations
	if(number>=2u) lu_solve(M, x, b, 2, 2);
	else lu_solve(M, x, b, 2, min(2u, number)); // cannot do loop unrolling here -> slower -> extra if-else to avoid slowdown
	const float A=x[0], H=x[1];
	const float K = 2.0f*A*cb(rsqrt(H*H+1.0f)); // mean curvature of Monge patch (x, f(x)), note that curvature definition in 2D is different than 3D (additional factor 2)
)+"#endif"+R( // D2Q9
	return clamp(K, -1.0f, 1.0f); // prevent extreme pressures in the case of almost degenerate matrices
}
)+"#endif"+R( // SURFACE

)+"#ifdef TEMPERATURE"+R(
)+R(void neighbors_temperature(const uxx n, uxx* j7) { // calculate neighbor indices
	uxx x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	j7[0] = n;
	j7[1] = xp+y0+z0; j7[2] = xm+y0+z0; // +00 -00
	j7[3] = x0+yp+z0; j7[4] = x0+ym+z0; // 0+0 0-0
	j7[5] = x0+y0+zp; j7[6] = x0+y0+zm; // 00+ 00-
}
)+R(void calculate_g_eq(const float T, const float ux, const float uy, const float uz, float* geq) { // calculate g_equilibrium from density and velocity field (perturbation method / DDF-shifting)
	const float wsT4=0.5f*T, wsTm1=0.125f*(T-1.0f); // 0.125f*T*4.0f (straight directions in D3Q7), wsTm1 is arithmetic optimization to minimize digit extinction, lattice speed of sound is 1/2 for D3Q7 and not 1/sqrt(3)
	geq[0] = fma(0.25f, T, -0.25f); // 000
	geq[1] = fma(wsT4, ux, wsTm1); geq[2] = fma(wsT4, -ux, wsTm1); // +00 -00, source: http://dx.doi.org/10.1016/j.ijheatmasstransfer.2009.11.014
	geq[3] = fma(wsT4, uy, wsTm1); geq[4] = fma(wsT4, -uy, wsTm1); // 0+0 0-0
	geq[5] = fma(wsT4, uz, wsTm1); geq[6] = fma(wsT4, -uz, wsTm1); // 00+ 00-
}
)+R(void load_g(const uxx n, float* ghn, const global fpxx* gi, const uxx* j7, const ulong t) {
	ghn[0] = load(gi, index_f(n, 0u)); // Esoteric-Pull
	for(uint i=1u; i<7u; i+=2u) {
		ghn[i   ] = load(gi, index_f(n    , t%2ul ? i    : i+1u));
		ghn[i+1u] = load(gi, index_f(j7[i], t%2ul ? i+1u : i   ));
	}
}
)+R(void store_g(const uxx n, const float* ghn, global fpxx* gi, const uxx* j7, const ulong t) {
	store(gi, index_f(n, 0u), ghn[0]); // Esoteric-Pull
	for(uint i=1u; i<7u; i+=2u) {
		store(gi, index_f(j7[i], t%2ul ? i+1u : i   ), ghn[i   ]);
		store(gi, index_f(n    , t%2ul ? i    : i+1u), ghn[i+1u]);
	}
}
)+"#endif"+R( // TEMPERATURE

)+R(void load_f(const uxx n, float* fhn, const global fpxx* fi, const uxx* j, const ulong t TS_P) { // TS_P = ", const global uint* tile_slot" bei SPARSE_TILES, sonst leer
)+"#ifdef SPARSE_TILES"+R( // Zellbasen zuerst sammeln: die abhaengigen tile_slot-Loads issuen dann zusammen statt seriell zu stallen
	const ulong T3 = (ulong)def_TILE*def_TILE*def_TILE;
	const ulong cbn = cell_base(n, tile_slot); // eigene Zelle -- einmal fuer alle 10 eigenen Zugriffe
	ulong cbj[def_velocity_set]; // Nachbarbasen vorab (nur ungerade i belegt)
	for(uint i=1u; i<def_velocity_set; i+=2u) cbj[i] = cell_base(j[i], tile_slot);
	fhn[0] = load(fi, cbn); // Esoteric-Pull (i=0 -> +0)
	for(uint i=1u; i<def_velocity_set; i+=2u) {
		fhn[i   ] = load(fi, cbn    + (ulong)(t%2ul ? i    : i+1u)*T3);
		fhn[i+1u] = load(fi, cbj[i] + (ulong)(t%2ul ? i+1u : i   )*T3);
	}
)+"#else"+R(
	fhn[0] = load(fi, index_f(n, 0u)); // Esoteric-Pull
	for(uint i=1u; i<def_velocity_set; i+=2u) {
		fhn[i   ] = load(fi, index_f(n   , t%2ul ? i    : i+1u));
		fhn[i+1u] = load(fi, index_f(j[i], t%2ul ? i+1u : i   ));
	}
)+"#endif"+R( // SPARSE_TILES
}
)+R(void store_f(const uxx n, const float* fhn, global fpxx* fi, const uxx* j, const ulong t TS_P) {
)+"#ifdef SPARSE_TILES"+R(
	const ulong T3 = (ulong)def_TILE*def_TILE*def_TILE;
	const ulong cbn = cell_base(n, tile_slot);
	ulong cbj[def_velocity_set];
	for(uint i=1u; i<def_velocity_set; i+=2u) cbj[i] = cell_base(j[i], tile_slot);
	store(fi, cbn, fhn[0]); // Esoteric-Pull (i=0 -> +0)
	for(uint i=1u; i<def_velocity_set; i+=2u) {
		store(fi, cbj[i] + (ulong)(t%2ul ? i+1u : i   )*T3, fhn[i   ]);
		store(fi, cbn    + (ulong)(t%2ul ? i    : i+1u)*T3, fhn[i+1u]);
	}
)+"#else"+R(
	store(fi, index_f(n, 0u), fhn[0]); // Esoteric-Pull
	for(uint i=1u; i<def_velocity_set; i+=2u) {
		store(fi, index_f(j[i], t%2ul ? i+1u : i   ), fhn[i   ]);
		store(fi, index_f(n   , t%2ul ? i    : i+1u), fhn[i+1u]);
	}
)+"#endif"+R( // SPARSE_TILES
}

)+"#ifdef SURFACE"+R(
)+R(void load_f_outgoing(const uxx n, float* fon, const global fpxx* fi, const uxx* j, const ulong t) { // load outgoing DDFs, even: 1:1 like stream-out odd, odd: 1:1 like stream-out even
	for(uint i=1u; i<def_velocity_set; i+=2u) { // Esoteric-Pull
		fon[i   ] = load(fi, index_f(j[i], t%2ul ? i    : i+1u));
		fon[i+1u] = load(fi, index_f(n   , t%2ul ? i+1u : i   ));
	}
}
)+R(void store_f_reconstructed(const uxx n, const float* fhn, global fpxx* fi, const uxx* j, const ulong t, const uchar* flagsj_su) { // store reconstructed gas DDFs, even: 1:1 like stream-in even, odd: 1:1 like stream-in odd
	for(uint i=1u; i<def_velocity_set; i+=2u) { // Esoteric-Pull
		if(flagsj_su[i+1u]==TYPE_G) store(fi, index_f(n   , t%2ul ? i    : i+1u), fhn[i   ]); // only store reconstructed gas DDFs to locations from which
		if(flagsj_su[i   ]==TYPE_G) store(fi, index_f(j[i], t%2ul ? i+1u : i   ), fhn[i+1u]); // they are going to be streamed in during next stream_collide()
	}
}
)+"#endif"+R( // SURFACE



)+R(kernel void initialize)+"("+R(global fpxx* fi, const global rhoxx* rho, global velxx* u, global uchar* flags // ) { // initialize LBM
)+"#ifdef SURFACE"+R(
	, global float* mass, global float* massex, global float* phi // argument order is important
)+"#endif"+R( // SURFACE
)+"#ifdef TEMPERATURE"+R(
	, global fpxx* gi, const global float* T // argument order is important
)+"#endif"+R( // TEMPERATURE
)+R( TS_P
)+") {"+R( // initialize()
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute initialize() on halo
)+"#ifdef SPARSE_TILES"+R(
	// FORK: Zellen in toten Tiles duerfen fi NIE anfassen. cell_base() gibt fuer sie defensiv 0
	// zurueck -- ohne diesen Ausstieg landen ihre Schreibzugriffe also in Slot 0 und ueberschreiben
	// die Daten einer echten aktiven Tile. Genau daran ist der erste T=8-Lauf divergiert (Cd 18.4).
	if(is_dead_tile(n, tile_slot)) return;
)+"#endif"+R(
	uchar flagsn = flags[n];
	const uchar flagsn_bo = flagsn&TYPE_BO; // extract boundary flags
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	uchar flagsj[def_velocity_set]; // cache neighbor flags for multiple readings
	for(uint i=1u; i<def_velocity_set; i++) flagsj[i] = flags[j[i]];
	if(flagsn_bo==TYPE_S) { // cell is solid
		bool TYPE_ONLY_S = true; // has only solid neighbors
		for(uint i=1u; i<def_velocity_set; i++) TYPE_ONLY_S = TYPE_ONLY_S&&(flagsj[i]&TYPE_BO)==TYPE_S;
		if(TYPE_ONLY_S) store3_u(u, n, (float3)(0.0f, 0.0f, 0.0f)); // reset velocity for solid lattice points with only boundary neighbors
)+"#ifndef MOVING_BOUNDARIES"+R(
		if(flagsn_bo==TYPE_S) store3_u(u, n, (float3)(0.0f, 0.0f, 0.0f)); // reset velocity for all solid lattice points
)+"#else"+R( // MOVING_BOUNDARIES
	} else if(flagsn_bo!=TYPE_E) { // local lattice point is not solid and not equilibrium boundary
		bool next_to_moving_boundary = false;
		for(uint i=1u; i<def_velocity_set; i++) {
			next_to_moving_boundary = next_to_moving_boundary||((flagsj[i]&TYPE_BO)==TYPE_S&&(load_u(u, j[i])!=0.0f||load_u(u, def_N+(ulong)j[i])!=0.0f||load_u(u, 2ul*def_N+(ulong)j[i])!=0.0f));
		}
		flags[n] = flagsn = next_to_moving_boundary ? flagsn|TYPE_MS : flagsn&~TYPE_MS; // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
)+"#endif"+R( // MOVING_BOUNDARIES
	}
	float feq[def_velocity_set]; // f_equilibrium
)+"#ifdef RHO_RAND"+R(
	{ const uxx rr_ = rr_idx(n); calculate_f_eq(rr_<(uxx)def_RR_N ? load_rho(rho, rr_) : 1.0f, load_u(u, n), load_u(u, def_N+(ulong)n), load_u(u, 2ul*def_N+(ulong)n), feq); } // ★ C2b: innen 1,0 = Saat rho_pack(1)
)+"#else"+R(
	calculate_f_eq(load_rho(rho, n), load_u(u, n), load_u(u, def_N+(ulong)n), load_u(u, 2ul*def_N+(ulong)n), feq);
)+"#endif"+R( // RHO_RAND
)+"#ifdef SURFACE"+R( // automatically generate the interface layer between fluid and gas
	{ // separate block to avoid variable name conflicts
		float phin = phi[n];
		if(!(flagsn&(TYPE_S|TYPE_E|TYPE_T|TYPE_F|TYPE_I))) flagsn = (flagsn&~TYPE_SU)|TYPE_G; // change all non-fluid and non-interface flags to gas
		if((flagsn&TYPE_SU)==TYPE_G) { // cell with updated flags is gas
			bool change = false; // check if cell has to be changed to interface
			for(uint i=1u; i<def_velocity_set; i++) change = change||(flagsj[i]&TYPE_SU)==TYPE_F; // if neighbor flag fluid is set, the cell must be interface
			if(change) { // create interface automatically if phi has not explicitely defined for the interface layer
				flagsn = (flagsn&~TYPE_SU)|TYPE_I; // cell must be interface
				phin = 0.5f;
				float rhon, uxn, uyn, uzn; // initialize interface cells with average density/velocity of fluid neighbors
				// ★ 12.09. abends (Pruefer A): SURFACE-Zweig. average_neighbors_fluid traegt den
				// float-Zeigertyp, u ist hier velxx -- unter -w meldet ocloc dazu nur "incompatible
				// pointer types" und baut. Heute unerreichbar (defines.hpp sperrt U_FP16 x SURFACE),
				// steht aber als Zwilling zum surface_0-Befund hier angesagt statt still.
				average_neighbors_fluid(n, rho, u, flags, &rhon, &uxn, &uyn, &uzn); // get average rho/u from all fluid neighbors
				calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
			}
		}
		if((flagsn&TYPE_SU)==TYPE_G) { // cell with updated flags is still gas
			store3_u(u, n, (float3)(0.0f, 0.0f, 0.0f)); // reset velocity for gas cells
			phin = 0.0f;
		} else if((flagsn&TYPE_SU)==TYPE_I && (phin<0.0f||phin>1.0f)) {
			phin = 0.5f; // cell should be interface, but phi was invalid
		} else if((flagsn&TYPE_SU)==TYPE_F) {
			phin = 1.0f;
		}
		phi[n] = phin;
		mass[n] = phin*load_rho(rho, n); // ★ 12.09.: der Typ-Zensus zaehlt SIGNATUREN, keine Ruempfe -- diese Zeile saehe er nicht
		massex[n] = 0.0f; // reset excess mass
		flags[n] = flagsn;
	}
)+"#endif"+R( // SURFACE
)+"#ifdef TEMPERATURE"+R(
	{ // separate block to avoid variable name conflicts
		float geq[7];
		calculate_g_eq(T[n], load_u(u, n), load_u(u, def_N+(ulong)n), load_u(u, 2ul*def_N+(ulong)n), geq);
		uxx j7[7]; // neighbors of D3Q7 subset
		neighbors_temperature(n, j7);
		store_g(n, geq, gi, j7, 1ul);
	}
)+"#endif"+R( // TEMPERATURE
	store_f(n, feq, fi, j, 1ul TS_A); // write to fi
} // initialize()

)+"#ifdef MOVING_BOUNDARIES"+R(
)+R(kernel void update_moving_boundaries(const global velxx* u, global uchar* flags) { // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute update_moving_boundaries() on halo
	const uchar flagsn = flags[n];
	const uchar flagsn_bo = flagsn&TYPE_BO; // extract boundary flags
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	uchar flagsj[def_velocity_set]; // cache neighbor flags for multiple readings
	for(uint i=1u; i<def_velocity_set; i++) flagsj[i] = flags[j[i]];
	if(flagsn_bo!=TYPE_S&&flagsn_bo!=TYPE_E&&!(flagsn&TYPE_T)) { // local lattice point is not solid and not equilibrium boundary and not temperature boundary
		bool next_to_moving_boundary = false;
		for(uint i=1u; i<def_velocity_set; i++) {
			next_to_moving_boundary = next_to_moving_boundary||((load_u(u, j[i])!=0.0f||load_u(u, def_N+(ulong)j[i])!=0.0f||load_u(u, 2ul*def_N+(ulong)j[i])!=0.0f)&&(flagsj[i]&TYPE_BO)==TYPE_S);
		}
		flags[n] = next_to_moving_boundary ? flagsn|TYPE_MS : flagsn&~TYPE_MS; // mark/unmark cells next to TYPE_S cells with velocity!=0 with TYPE_MS
	}
} // update_moving_boundaries()
)+"#endif"+R( // MOVING_BOUNDARIES



)+"#ifdef REGULARIZED_BOUNDARIES"+R(
)+R(float deriv_reg(const global velxx* u, const ulong off, const uxx jp, const uxx jm, const bool fp, const bool fm, const float u0) {
	// Ableitung einer Geschwindigkeitskomponente entlang einer Achse, aus dem FELD u[].
	// Zentral, wenn beide Nachbarn echtes Fluid sind; sonst einseitig; sonst null.
	// Als Funktion und nicht als Makro: #define innerhalb des Kernel-Strings wuerde vom
	// Host-Praeprozessor verschluckt, nicht vom OpenCL-Uebersetzer.
	return fp&&fm ? 0.5f*(load_u(u, off+(ulong)jp)-load_u(u, off+(ulong)jm)) : (fp ? load_u(u, off+(ulong)jp)-u0 : (fm ? u0-load_u(u, off+(ulong)jm) : 0.0f));
} // deriv_reg()

)+R(float reg_fneq(const uint i, const float regf, const float Sxx, const float Syy, const float Szz, const float Sxy, const float Sxz, const float Syz, const float trS3) {
	// f_neq_i = regf * w_i * (c_i . S . c_i - tr(S)/3) mit regf = -3*rho/w.
	// Als KOMPAKTE Funktion statt des frueheren Riesen-Makros, das den vollen Ausdruck 19-fach in
	// die ternaere Kollisionszeile expandierte. Zwei Fassungen davon haben den Intel-Uebersetzer
	// aufgehaengt -- die zweite hat den ganzen Rechner eingefroren, weil der Desktop auf derselben
	// GPU laeuft. Uebersetzerfreundlichkeit ist hier also Betriebssicherheit, kein Stil.
	const float cxi=c(i), cyi=c(def_velocity_set+i), czi=c(2u*def_velocity_set+i);
	const float cSc = fma(cxi*cxi, Sxx, fma(cyi*cyi, Syy, fma(czi*czi, Szz, 2.0f*fma(cxi*cyi, Sxy, fma(cxi*czi, Sxz, cyi*czi*Syz)))));
	const float wi = (i==0u) ? def_w0 : ((i<=6u) ? def_ws : def_we);
	return regf*wi*(cSc-trS3);
} // reg_fneq()
)+"#endif"+R( // REGULARIZED_BOUNDARIES

)+"#if defined(WANDFUNKTION)||defined(FACETTEN)"+R(
float wf_spalding_uplus(const float Y) {
	// Loese X*S(X) = Y fuer X = u+ (Spalding, kappa=0,41, B=5,5 wie Han et al. 2021, Gl. 16).
	// Log-Log-Newton, Start sqrt(Y), FIX drei Iterationen ohne Konvergenzabfrage (Branch-Divergenz).
	// Genauigkeit (Messung 2026-08-19 MIT korrekter Konstante, FP32 gegen double-Bisektion, it=3):
	// tau_w-Fehler -0,44 % bei Y~2400, -4,4 % bei Y=1e4, <1e-4 bei Y<=330 (Fahrzeugbereich);
	// bei hohem Re_tau Iterationszahl erhoehen. (Die frueheren "R2"-Zahlen ~0,1 %/2,3 % galten
	// fuer die FALSCHE Konstante und sind hinfaellig -- R3-Etikettenbereinigung.) X geklemmt auf <=100: kappa*X <= 41, exp davon ist
	// FP32-sicher -- noetig, weil -cl-finite-math-only NaN/INF zu undefiniertem Verhalten macht.
	// ★ GROSS-AUDIT HOCH (2026-08-19, Pruefer 1): hier stand 0.010517092f -- das ist (e^0.1 - 1)/10,
	// ein Uebertragungsfehler, effektiv B = 11,11 statt 5,5. Folge: u+ zu gross, tau_w = rho*(ut/u+)^2
	// systematisch 16-38 % zu klein in ALLEN Wandmodell-Pfaden (WFB, Paararm, iMEM, PEMA) seit dem
	// WFB-Bau; erklaert einen Grossteil des -68-%-Kanalbefunds. Die dokumentierte "1e-13"-Genauigkeit
	// war Bisektion DERSELBEN Gleichung mit DERSELBEN Konstante -- selbstkonsistent, blind dafuer.
)+"#ifdef SPALDING_TAB"+R(
	// ★ 11.09.2026: Tabellennachschlag statt drei Newton-Schritten (CFD_SPALDING_TAB=1).
	// Stuetzstellen log-gleich in Y, abgelegt ist log(u+), linear interpoliert. Beide Achsen
	// logarithmisch, dort ist die Kurve fast gerade -- gemessen max 0,0035 % tau_w-Fehler
	// gegen 4,364 % bei it=3. Die Tabelle ist in __constant abgelegt, nicht privat: ein
	// laufzeitindiziertes privates Array waere die Scratch-Falle.
	float tl = (log(fmax(Y, 1e-12f))-def_spald_l0)*def_spald_invdl;
	tl = clamp(tl, 0.0f, def_spald_max);
	const uint i0 = (uint)tl;
	const float fr = tl-(float)i0;
	return fmin(100.0f, exp(def_spald_tab[i0]+fr*(def_spald_tab[i0+1u]-def_spald_tab[i0])));
)+"#else"+R(
	const float kap=0.41f, emkB=0.104874f; // exp(-kappa*B) mit B=5,5 = 0.1048735
	float x = log(fmin(100.0f, sqrt(fmax(Y, 1e-12f))));
	for(uint it=0u; it<def_wf_spalding_it; it++) { // Iterationszahl als Define (Stufe-2-Auflage 8; Default 3)
		const float X = fmin(100.0f, exp(x));
		const float kX = kap*X, ekX = exp(kX);
		const float S  = X + emkB*(ekX-1.0f-kX-0.5f*kX*kX-0.16666667f*kX*kX*kX);
		const float Sp = 1.0f + emkB*kap*(ekX-1.0f-kX-0.5f*kX*kX);
		const float g  = log(fmax(X*S, 1e-30f)) - log(fmax(Y, 1e-30f));
		const float gp = 1.0f + X*Sp/S;
		x -= g/gp;
	}
	return fmin(100.0f, exp(x));
)+"#endif"+R( // SPALDING_TAB
} // wf_spalding_uplus()
)+"#endif"+R( // WANDFUNKTION||FACETTEN
)+"#ifdef WANDFUNKTION"+R(
void apply_wall_function(float* fhn, const uxx* j, const global uchar* flags, global uint* wf_hits, const ulong t, const bool zaehle) {
	// ★★ WANDFUNKTIONS-BOUNCE-BACK nach Han/Ooka/Kikumoto 2021 (Fluid Dyn. Res. 53, 045506), fuer
	// EBENE z-WAENDE (der Kanal; das Fahrzeug folgt mit den zellbasierten Facetten, C1b).
	//
	// KEIN "Bounce-Back plus Korrektur", sondern TAUSCH plus Abzug -- und das ist der ganze Punkt:
	// an einer gitterparallelen Wand haengt der GESAMTE tangentiale BB-Widerstand an den vier
	// Diagonallinks (der senkrechte traegt nichts, siehe audit_bewegte_waende). Die WFB ersetzt
	// genau diese vier: erst die FREE-SLIP-Spiegelung (Tausch der beiden von der Wand einlaufenden
	// Diagonalpartner -- entfernt den rauwandartigen BB-Widerstand), dann der Abzug von exakt
	// (1/2)*tau_w je Link (setzt den modellierten Widerstand an dessen Stelle). Wer stattdessen auf
	// den BB ADDIERT, doppelzaehlt -- V1s teuerste Fehlerklasse.
	// Masse: Tausch + antisymmetrisches +- ist exakt erhaltend. Normalimpuls: unberuehrt.
	// def_wf_tau = 0 ist der Zwischenarm "nur Tausch" (reiner Free-Slip, Schritt 4 des Plans).
	const bool boden = (flags[j[6]]&TYPE_BO)==TYPE_S; // Solid unter der Zelle (-z)
	const bool decke = (flags[j[5]]&TYPE_BO)==TYPE_S; // Solid ueber der Zelle (+z)
	if(!boden&&!decke) return;
	if(boden&&decke) { if(zaehle&&t%def_zaehl_takt==0ul) atomic_inc(&wf_hits[5]); return; } // Spalt von 1 Zelle: unbehandelt, aber GEZAEHLT
	float rhon, uxn, uyn, uzn;
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // Zustand VOR der Korrektur (Hans Abtastpunkt, y = 0,5)
	const float ut = sqrt(uxn*uxn+uyn*uyn);
	float tau_x = 0.0f, tau_y = 0.0f; // WM-Blick B: bei achse==2 wird utz nie angewandt -- dominante-Achsen-Naeherung des KONTROLLARMS (iMEM traegt volles 3D-u_s)
	if(ut>=1e-6f) {
		const float Y  = ut*def_wf_Y; // = |u_t| * 0,5/nu
		const float up = wf_spalding_uplus(Y);
		const float utau = ut/up;
		float tw = rhon*utau*utau;
		const float tw_max = 0.5f*rhon*ut; // physikalische Klemme; Treffer werden gezaehlt
		if(tw>tw_max) { tw = tw_max; if(zaehle&&t%def_zaehl_takt==0ul) atomic_inc(&wf_hits[3]); } // Slot 3: NUR tau-Klemme
		tau_x = -def_wf_tau*tw*uxn/ut; // Widerstand GEGEN u_t (Han Gl. 15)
		tau_y = -def_wf_tau*tw*uyn/ut;
	} else if(zaehle&&t%def_zaehl_takt==0ul) atomic_inc(&wf_hits[4]); // Slot 4: u_t~0-Skips (Audit: vorher mit Slot 3 vermischt)
	if(boden) { // einlaufende Diagonalen: 9=(+1,0,+1)/16=(-1,0,+1) und 11=(0,+1,+1)/18=(0,-1,+1)
		const float a=fhn[9]; fhn[9]=fhn[16]+0.5f*tau_x; fhn[16]=a-0.5f*tau_x;
		const float b=fhn[11]; fhn[11]=fhn[18]+0.5f*tau_y; fhn[18]=b-0.5f*tau_y;
	} else { // Decke, einlaufend: 15=(+1,0,-1)/10=(-1,0,-1) und 17=(0,+1,-1)/12=(0,-1,-1)
		const float a=fhn[15]; fhn[15]=fhn[10]+0.5f*tau_x; fhn[10]=a-0.5f*tau_x;
		const float b=fhn[17]; fhn[17]=fhn[12]+0.5f*tau_y; fhn[12]=b-0.5f*tau_y;
	}
	if(zaehle&&t%def_zaehl_takt==0ul) atomic_inc(&wf_hits[2]); // Wirkpfad-Zaehler, gegatet gegen uint-Ueberlauf
} // apply_wall_function()
)+"#endif"+R( // WANDFUNKTION
)+"#ifdef FACETTEN"+R(
// ★★ C1b Stufe 2: Facetten-WFB, dominante Achse (FACETTEN-STUFE2.md). Verallgemeinert die z-WFB
// auf alle 6 Wandseiten. ZWEI Pflichten aus der Plan-Revision: (R1) jedes Diagonalpaar wird NUR
// getauscht, wenn BEIDE Streaming-Urspruenge solid sind -- an Treppen zeigt sonst ein Link ins
// Fluid und der Tausch zerstoert regulaer gestreamte Information (der Kanal war nur per
// Geometriezufall sicher); (R2) der ANGEWANDTE tau-Anteil traegt den Flaechenfaktor 1/|n_achse|
// (host-vorberechnet, am parallelen Kanal exakt 1,0), sonst fehlen bei 45 Grad 29,3 % des
// integrierten Widerstands. Der AKKUMULATOR bekommt das PHYSIKALISCHE tau_w (ohne Faktor), denn
// daraus wird y+ gebildet -- und nur von Zellen, die wirklich >=1 Paar getauscht haben (Auflage 3).
// Ausdrucksbaum bei n=ez wortgleich zur z-WFB -- der Kanal-Aequivalenznachweis kollabiert bitgenau
// (2.0f*0.5f==1.0f, 1.0f*def_fac_Y==def_fac_Y, faca==1.0f => fmin(tw*1.0f,twmax)==tw).
// Ein Tauschpaar mit linkweisem Gate (R1): nur wenn BEIDE Streaming-Urspruenge solid sind.
// Gate-Maske wie z-WFB: (flags&TYPE_BO)==TYPE_S schliesst TYPE_E und TYPE_MS linkweise aus.
// Indizes sind an allen Aufrufstellen Literale -- der Uebersetzer inlinet ohne dynamische fhn-Indizierung.
void fac_paar(float* fhn, const global uchar* flags, const uxx* j, const uint ip, const uint im,
              const uint g1, const uint g2, const float taut, uint* getauscht, float* fkraft) {
	if(((flags[j[g1]]&TYPE_BO)==TYPE_S)&&((flags[j[g2]]&TYPE_BO)==TYPE_S)) {
		const float fp_=fhn[ip]; fhn[ip]=fhn[im]+0.5f*taut; fhn[im]=fp_-0.5f*taut; (*getauscht)++;
		*fkraft -= taut; // Wandkraft der ANGEWANDTEN Korrektur (Cd-Pfad E5): -tau_t = +def_fac_tau*twe*ut_c/ut
	}
}
void apply_facette(const uxx n, float* fhn, const uxx* j, const global uchar* flags,
                   const global float* fac_geo, const global uint* fac_idx,
                   global float* fac_tau_acc, global uint* fac_tau_cnt, global uint* hits, const ulong t) {
	uxx fbi; if(!f_bbox(n, &fbi)) return;      // ausserhalb der F-BBox gibt es keine Facetten
	const uint fid = fac_fid(fac_idx, fbi);
	if(fid==0xFFFFFFFFu) return;               // keine oder markierte Facette: reiner BB
	const uxx b = 8ul*(uxx)fid;
	const float nx=fac_geo[b], ny=fac_geo[b+1ul], nz=fac_geo[b+2ul], yw=fac_geo[b+3ul], faca=fac_geo[b+4ul];
	const uint achse = (uint)fac_geo[b+5ul];
	float rhon, uxn, uyn, uzn;
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // Zustand VOR der Korrektur (Hans Abtastpunkt)
	const float und = nx*uxn+ny*uyn+nz*uzn;
	const float utx=uxn-und*nx, uty=uyn-und*ny, utz=uzn-und*nz; // voller 3D-Tangentialvektor
	const float ut = sqrt(utx*utx+uty*uty+utz*utz);
	float tw=0.0f, twe=0.0f; // tw = physikalisch (Akkumulator), twe = angewandt (mit Flaechenfaktor)
	if(ut>=1e-6f) {
		const float Y  = ut*((2.0f*yw)*def_fac_Y); // = |u_t| * y_w/nu; bei yw=0,5 bitgenau ut*def_wf_Y
		const float up = wf_spalding_uplus(Y);
		const float utau = ut/up;
		tw = rhon*utau*utau;
		const float tw_max = 0.5f*rhon*ut;
		if(tw>tw_max) { tw = tw_max; if(t%def_zaehl_takt==0ul) atomic_inc(&hits[8]); } // Slot 8: Facetten-tau-Klemme (gegatet t%100 -- Audit R3: Fahrzeugmassstab wickelte ungegatet ueber)
		const float twf = tw*faca; // R2-Flaechenfaktor
		if(twf>tw_max&&t%def_zaehl_takt==0ul) atomic_inc(&hits[8]); // Zweitklemme, Slot 8 = beide (gegatet t%100 seit R3)
		twe = fmin(twf, tw_max); // am Kanal faca==1: twf==tw, nach Erstklemme nie >tw_max -> bitgleich und zaehlerneutral
	} else if(t%def_zaehl_takt==0ul) atomic_inc(&hits[9]); // Slot 9: u_t~0-Skip, gegatet t%100 (Tausch passiert trotzdem, wie z-WFB)
	uint getauscht = 0u; float fk_x=0.0f, fk_y=0.0f, fk_z=0.0f; // angewandte Wandkraft je Komponente (Cd-Pfad)
	// Paartabelle FACETTEN-STUFE2.md Abschnitt B; Gate-Maske wie z-WFB: (flags&TYPE_BO)==TYPE_S
	// schliesst TYPE_E und TYPE_MS (0x03) linkweise aus. Paarreihenfolge: tangentiale Achsen aufsteigend.
	if(achse==2u) {
		const float tau_x = (ut>=1e-6f) ? -def_fac_tau*twe*utx/ut : 0.0f;
		const float tau_y = (ut>=1e-6f) ? -def_fac_tau*twe*uty/ut : 0.0f;
		if(nz>0.0f) { // Boden (-z): einlaufend 9=(+1,0,+1)/16=(-1,0,+1), 11=(0,+1,+1)/18=(0,-1,+1)
			fac_paar(fhn, flags, j, 9u, 16u, 10u, 15u, tau_x, &getauscht, &fk_x);
			fac_paar(fhn, flags, j, 11u, 18u, 12u, 17u, tau_y, &getauscht, &fk_y);
		} else {      // Decke (+z): 15=(+1,0,-1)/10=(-1,0,-1), 17=(0,+1,-1)/12=(0,-1,-1)
			fac_paar(fhn, flags, j, 15u, 10u, 16u, 9u, tau_x, &getauscht, &fk_x);
			fac_paar(fhn, flags, j, 17u, 12u, 18u, 11u, tau_y, &getauscht, &fk_y);
		}
	} else if(achse==0u) {
		const float tau_y = (ut>=1e-6f) ? -def_fac_tau*twe*uty/ut : 0.0f;
		const float tau_z = (ut>=1e-6f) ? -def_fac_tau*twe*utz/ut : 0.0f;
		if(nx>0.0f) { // Wand bei -x: einlaufend 7=(+1,+1,0)/13=(+1,-1,0), 9=(+1,0,+1)/15=(+1,0,-1)
			fac_paar(fhn, flags, j, 7u, 13u, 8u, 14u, tau_y, &getauscht, &fk_y);
			fac_paar(fhn, flags, j, 9u, 15u, 10u, 16u, tau_z, &getauscht, &fk_z);
		} else {      // Wand bei +x: 14=(-1,+1,0)/8=(-1,-1,0), 16=(-1,0,+1)/10=(-1,0,-1)
			fac_paar(fhn, flags, j, 14u, 8u, 13u, 7u, tau_y, &getauscht, &fk_y);
			fac_paar(fhn, flags, j, 16u, 10u, 15u, 9u, tau_z, &getauscht, &fk_z);
		}
	} else {
		const float tau_x = (ut>=1e-6f) ? -def_fac_tau*twe*utx/ut : 0.0f;
		const float tau_z = (ut>=1e-6f) ? -def_fac_tau*twe*utz/ut : 0.0f;
		if(ny>0.0f) { // Wand bei -y: einlaufend 7=(+1,+1,0)/14=(-1,+1,0), 11=(0,+1,+1)/17=(0,+1,-1)
			fac_paar(fhn, flags, j, 7u, 14u, 8u, 13u, tau_x, &getauscht, &fk_x);
			fac_paar(fhn, flags, j, 11u, 17u, 12u, 18u, tau_z, &getauscht, &fk_z);
		} else {      // Wand bei +y: 13=(+1,-1,0)/8=(-1,-1,0), 18=(0,-1,+1)/12=(0,-1,-1)
			fac_paar(fhn, flags, j, 13u, 8u, 14u, 7u, tau_x, &getauscht, &fk_x);
			fac_paar(fhn, flags, j, 18u, 12u, 17u, 11u, tau_z, &getauscht, &fk_z);
		}
	}
	if(getauscht>0u) { // 1 Zelle = 1 Facette: kein Atomic noetig; Layout 6 float (iMEM-Umbau): [0] tw physisch (y+), [1..3] angewandte Wandkraft (Cd-Reibung), [4] Delta-m (Paararm 0), [5] Normalkontamination (Paararm 0)
		fac_tau_acc[6ul*(uxx)fid] += tw; fac_tau_acc[6ul*(uxx)fid+1ul] += fk_x; fac_tau_acc[6ul*(uxx)fid+2ul] += fk_y; fac_tau_acc[6ul*(uxx)fid+3ul] += fk_z;
		fac_tau_cnt[fid] += 1u; }
	else if(t%def_zaehl_takt==0ul) atomic_inc(&hits[11]); // Slot 11: Facette da, aber kein Paar offen (gegatet)
	if(t%def_zaehl_takt==0ul) atomic_inc(&hits[7]);       // Slot 7: Wirkpfad, Soll = N_aktiv * ceil(n_steps/100)
} // apply_facette()
)+"#endif"+R( // FACETTEN
)+"#ifdef FACETTEN_IMEM"+R(
)+"#ifdef FACETTEN_ELIBB"+R(
float3 elibb_rekonstruiere(float* fhn, const uxx* j, const global uchar* flags, const global uchar* fac_q, const uint fid,
                         const float rhon, const float upx, const float upy, const float upz,
                         const float fnx, const float fny, const float fnz,
                         global float* fac_tau_acc, global uint* hits, const ulong t) {
	// ★ PERF-FIX 2026-08-26 (IGC-Dump-Diagnose): der fruehere dp_out-Pointer machte das
	// Caller-Array elibb_dp[3] adressierbar -- IGC legte den ELIBB-Pfad mit private_size=4256 B
	// je Work-Item in den Speicher (Scratch-Signatur: Faktor ~100, 0 GB/s). Rueckgabe jetzt
	// PER WERT als float3 (registerfaehig, keine Adresse). Diagnose: externes Review sagte
	// die Klasse exakt voraus; zeinfo-Beleg im Scratchpad igc0/igc1.
	// ★★ B2 REVISION nach Pruefbefund W2 (2026-08-25): die Blende ist REIN GEOMETRISCH (u_W = 0).
	// Die erste Fassung trug u_W = u_s IN der Blende -- bei q = 0,5 ist die Blende aber die
	// Identitaet, der Wandmodell-Impuls waere an JEDER ebenen Partie (q = 0,5: Kanalboden,
	// Unterboden) konstruktiv ausgefallen. Jetzt: Blende verschiebt NUR den Reflexionspunkt auf
	// die Facettenebene; den Wandmodell-Impuls traegt weiterhin der bestehende Additivterm
	// q_i = 6 w_i (c_i . u_s) in Pass 2 (samt alpha-Massenkorrektur -- deren Mathematik haengt nur
	// am Additivterm und bleibt exakt). Kein Stapeln: die Blende traegt KEIN u_W mehr.
	// Bei q = 0,5 kollabiert der gesamte Pfad BITGLEICH auf das heutige iMEM -- der ebene
	// Kanal-Anker prueft damit wieder echte Bitgleichheit (qb==127-Kurzschluss, auch -0.0-fest).
	// NEBB in STOERFORM (FP16-Ablage: f^ = f - w; w-Subtraktion ist Pflicht -- P1-Offset-Falle;
	// ★ 21.09.2026 berichtigt: hier stand "FP16C", gebaut ist aber FP16S (defines.hpp:27, FP16C:80
	// auskommentiert). Die Stoerform gilt fuer BEIDE 2-Byte-Formate, der Name war falsch.):
	//   f~_i = w_i*(rho - 1) + (f_ib - feq_ib(u_pre)),  feq_ib mit c_ib = -c_i.
	// SNAPSHOT ALLER 19 (V1-Audit-Fix A, 1-Zell-Spalt-Aliasing). Slotlogik: Harness A/B (B0).
	float fpre[def_velocity_set];
	for(uint i=0u; i<def_velocity_set; i++) fpre[i]=fhn[i];
	const float up2 = upx*upx+upy*upy+upz*upz;
	// ★★ TANGENTIALPROJEKTION von u_pre (Kernel-Audit Befund 1, 2026-08-25 nacht; speist seit dem
	// MLS-Umbau 26.08. den MLS-Zweig unten = dessen Interim I2). Der Couette-Fixpunkt-Harness des
	// Pruefagenten zeigte damals am K1'-Zweig: feq_ib mit der WANDNORMAL-Komponente von u_pre
	// koppelt im kohaerenten Fall (ebene Panels, einheitliches q>0,5 -- exakt die m2-Klasse, 97 %
	// am Fahrzeug) positiv zurueck und DIVERGIERT (qb=128: ~4200 Schritte, q=0,6: 88). Mit
	// Projektion: stabil UND wandlagen-treu (-0,51->-0,510, -1,0->-0,875).
	// Der q<0,5-Zweig behaelt das VOLLE u_pre (R6: unveraendert, als sauber vermessen).
	const float upn_ = fnx*upx+fny*upy+fnz*upz;
	const float uptx = upx-upn_*fnx, upty = upy-upn_*fny, uptz = upz-upn_*fnz;
	const float upt2 = uptx*uptx+upty*upty+uptz*uptz;
	bool beruehrt = false;
	// ★ PERF-FIX 2 (2026-08-26, offline per ocloc/zeinfo bewiesen): ohne Unroll-Hint rollt IGC diese
	// gewachsene Schleife nicht mehr aus, die privaten Richtungs-Arrays hinter c()/w() plus fpre[]
	// werden speicherheimisch -> private_size 4256 B/WI (iGPU) bzw. 8512 B (B70) = Scratch = 100x-Bremse
	// (2 statt 240 MLUPs, g13-g15). MIT Hint: private_size 0 auf beiden Geraeten (Varianten H/H2, igc3).
	// Quelltext-identische Operationen je Iteration; der MASSGEBLICHE Bitgleichheits-Beweis ist die
	// Messung (g16: FELD-HASH beider Arme = kipp0-Anker 4722579264326613690) -- ein Compiler DARF
	// gerollt/ausgerollt unterschiedlich kontrahieren (-cl-mad-enable), tut es hier belegt nicht.
	// (18 = D3Q19-Linkzahl; bei einem D3Q27-Umbau auf 26 anheben -- Auditor-B-Hinweis.)
	__attribute__((opencl_unroll_hint(18)))
	for(uint i=1u; i<def_velocity_set; i++) {
		const uint ib = (i%2u==1u) ? i+1u : i-1u;               // Streaming-Ursprung von fhn[i] ist j[ib]
		if((flags[j[ib]]&TYPE_BO)!=TYPE_S) continue;             // nur wandstaemmige Links
		const uchar qb = fac_q[18ul*(uxx)fid+(uxx)(ib-1u)];      // q gehoert zum Link ib (Harness B)
		if(qb==0u) continue;                                     // Ebene schneidet den Link nicht -> implizites BB bleibt
		if(qb==127u) continue;                                   // q = 0,5: Blende ist Identitaet -- Kurzschluss haelt auch die -0.0-Kante bitgleich
		const float q = (float)qb*(1.0f/254.0f);
		const float cix=c(i), ciy=c(def_velocity_set+i), ciz=c(2u*def_velocity_set+i);
		const float wi=w(i);
		const float cup = -(cix*upx+ciy*upy+ciz*upz);            // c_ib . u_pre  (c_ib = -c_i)
		const float feq_ib = wi*(rhon*(1.0f+3.0f*cup+4.5f*cup*cup-1.5f*up2)-1.0f); // feq_ib(rho,u_pre), Stoerform
		const float bb = fpre[i];
		if(q<=0.5f) { // Zweig unveraendert (R6: eine Variable je Schritt; QDIAG=2 hat ihn als sauber vermessen)
			const float nebb = wi*(rhon-1.0f) + (fpre[ib]-feq_ib); // f~ = w(rho-1) + f_neq_ib -- u_W = 0
			fhn[i] = fma(2.0f*q, bb, (1.0f-2.0f*q)*nebb);
		} else {
			// ★★ MLS-Blende (Physik-Kette Baustein 1, 2026-08-26): ersetzt den K1'-Zweig. K1' trug
			// den wandlokalen Tangential-Geistmode lambda*3*w*rho*(c.u_t) mit lambda=(2q-1)/q und
			// Neutralkurve lambda_krit=4(2-omega)/(omega-1) -- bei omega0~1,9999 ist q_krit~0,51,
			// d.h. praktisch JEDES q>0,5 instabil; die QKAPPE=0,65-Kappe war selbst instabil
			// (~25k Schritte) und nur durch Smagorinsky-nu_t maskiert (Herleitung/Historie inkl.
			// Eq.-25-Aufloesung: Planungsbericht zu Commit 75aad52, Wissensspeicher k1instabilitaet).
			// ERSATZ: chi-Blende nach Mei/Shyy/Yu/Luo. ZITAT-PRAEZISION (Literatur-Verifikation
			// 26.08.2026, visuell an den NASA/ICASE-Drucken): die q>=1/2-Form chi=(2q-1)/(tau+1/2),
			// u_bf=(1-3/(2q))*u_f steht NICHT in Mei-Luo-Shyy JCP 155 (1999) 307 (dort behaelt
			// Abschn. 2.2 fuer q>=1/2 den FH-Zweig chi=(2q-1)/tau), sondern erst in JCP 161 (2000)
			// 680, Gl. (2.4) der Reportfassung ICASE 2002-17, und PRE 65, 041203 (2002), Gl.
			// (2.1)-(2.3b). Blend (Gl. 1.9, u_w=0 -- Facetten ruhen, kein u_w-Term):
			//   fhn[i] = (1-chi)*bb + chi*f*,  f* = w*(rho*(1+3(c_ib.u_bf)+4,5(c_ib.u_f)^2-1,5|u_f|^2)-1)
			// Stoerform exakt (linear in f). Quadratische Terme tragen u_f, NUR der lineare u_bf
			// (drei Drucke einig). c_ib = -c_i = Richtung Fluid->Wand wie im Vorgaenger (die
			// Minuszeichen in cupt/cub); ein Richtungsfehler zeigt sich NUR im linearen Term und
			// tarnt sich bei u_w=0 als Fehlskalierung -> Wandlage-Test in der S0-Abnahme.
			// q=0,5 => chi=0 => fhn[i]=bb BITGLEICH (Anker; qb==127-Kurzschluss oben bleibt).
			// chi<=1 fuer q<=1 (Konvexblend), tau>=0,5 -- stabil im Harness bis omega=1,999, q=1.
			// ZWEI DEKLARIERTE INTERIMS (Keine-Handwerte, je mit Abloesebedingung):
			//  (I1) tau in chi ist tau0 (JIT-Konstante def_fac_chifak=1/(tau0+0,5), lbm.cpp), nicht
			//       das lokale SUBGRID-tau_eff; Chapman-Enskog bindet chi ans Kollisions-tau.
			//       Konservativ: tau_eff>=tau0 => chi_eff<=chi -> tiefer im Konvexbereich.
			//       Abloesung: lokales tau_eff durchreichen, sobald der Facettenpfad es kennt.
			//  (I2) u_f geht TANGENTIAL PROJIZIERT ein (upt statt up) -- Abweichung vom Druck
			//       (dort volles u_f); Grund: Kernel-Audit Befund 1 (Couette-Fixpunkt: der
			//       Wandnormalanteil koppelt im kohaerenten Panel-Fall positiv zurueck).
			//       Abloesung: A/B upt vs. up auf der 8mm-Sprosse nach validierter MLS-Basis.
			const float chi = (2.0f*q-1.0f)*def_fac_chifak;   // chi = (2q-1)/(tau0+0,5)
			const float cupt = -(cix*uptx+ciy*upty+ciz*uptz); // c_ib . u_f (quadratischer Term)
			const float cub = (1.0f-1.5f/q)*cupt;             // c_ib . u_bf, u_bf = (1-3/(2q))*u_f
			const float fst = wi*(rhon*(1.0f+3.0f*cub+4.5f*cupt*cupt-1.5f*upt2)-1.0f); // f* (Stoerform)
			fhn[i] = fma(1.0f-chi, bb, chi*fst);
			// Eigener Wirkpfad des MLS-Zweigs (Frische-Augen-Audit 26.08., Befund 1): Slot 67
			// zaehlt BEIDE Zweige (beruehrt), erst dieser Zaehler beweist im Binary, dass der
			// q>0,5-Zweig Zellen anfasst. Muster Slot 67: saettigend, t%100-gegatet.
			if(t%def_zaehl_takt==0ul&&hits[68]<0xF0000000u) atomic_inc(&hits[68]);
		}
		beruehrt = true;
	}
	if(beruehrt) {
		// ★★ B3 (2026-08-25 nacht): der Blenden-Austausch wird GEBUCHT. Die Blende aendert nur den
		// EINLAUF -- ihr Wandimpuls ist DeltaF_Wand = -Sum c_i*(fhn_neu - fpre) in EINFACHzaehlung
		// (kein MEM-2x; das 2x der phi-Buchung gilt fuer den BB-Rundlauf, nicht fuer eine reine
		// Einlaufmodifikation). Masse: Dm = Sum(fhn_neu - fpre) -> Slot [4], vom bestehenden
		// Delta-m-Waechter mitueberwacht. Bucht NUR wenn beruehrt (kipp0-Bitanker: qb=127 ->
		// unberuehrt -> Akkumulator bit-unangetastet). Damit sind Reibungspfad und object_force
		// wieder EIN Bild -- die g12-Cz-Pfad-Diskrepanz war der Preis der fehlenden Buchung.
		float dpx=0.0f, dpy=0.0f, dpz=0.0f, dm_=0.0f;
		for(uint i=1u; i<def_velocity_set; i++) {
			const float d_ = fhn[i]-fpre[i];
			if(d_==0.0f) continue;
			dpx = fma(d_, c(i), dpx); dpy = fma(d_, c(def_velocity_set+i), dpy); dpz = fma(d_, c(2u*def_velocity_set+i), dpz);
			dm_ += d_;
		}
		const uxx a3 = 6ul*(uxx)fid;
		// racefrei OHNE Atomics: 1 Zelle = 1 Facette (bijektiv, s. Paararm-Kommentar oben und den
		// Invarianten-Waechter in alloc_facetten_domain) -- Audit-Entwarnung 2026-08-26.
		fac_tau_acc[a3+1ul] += -dpx; fac_tau_acc[a3+2ul] += -dpy; fac_tau_acc[a3+3ul] += -dpz;
		fac_tau_acc[a3+4ul] += dm_;
		// dp geht per Wert zurueck (B3-Pruefbefund 1b: +2*Dp_t-Korrektur an der phi-Buchung)
		if(t%def_zaehl_takt==0ul&&hits[67]<0xF0000000u) atomic_inc(&hits[67]); // Wirkpfad, saettigend
		return (float3)(dpx, dpy, dpz);
	}
	return (float3)(0.0f, 0.0f, 0.0f);
} // elibb_rekonstruiere()
)+"#endif"+R( // FACETTEN_ELIBB
)+"#ifdef FAC_R1Q_AN"+R(
// ★ 28.09.2026 R1Q: masselose Zellquelle Sum c_i Df_i = rho*du in der Delta-Form des Gleichgewichts
// (dieselbe Form wie der FAC_REK-Block: Df = f_eq(rho,u+du) - f_eq(rho,u), s = 2u + du). Eigene Funktion,
// damit der REK-Block Zeichen fuer Zeichen unveraendert bleibt. Aufruf nur weiter unten (C99).
float4 r1q_einspeisen(float* fhn, const float rhon, const float uxn, const float uyn, const float uzn, const float dux, const float duy, const float duz) {
	// ★ Rueckgabe: (Sum c_x Df, Sum c_y Df, Sum c_z Df, Sum Df), gebildet aus den INKREMENTEN selbst -- nicht als
	// Differenz zweier fp32-Summen ueber fhn (deren Rundung bis ~3e-6*Sum|f| ueberdeckte jeden Fehler dieser Groesse;
	// CPU-Stufe 28.09.: Slot 429 feuerte an 0,3 % bei korrektem Code). So sind die Proben 428/429 streng.
	const float sx = fma(2.0f, uxn, dux);
	const float sy = fma(2.0f, uyn, duy);
	const float sz = fma(2.0f, uzn, duz);
	const float h3 = -1.5f*fma(sx, dux, fma(sy, duy, sz*duz));
	const float wr_s = def_ws*rhon;
	const float wr_e = def_we*rhon;
	const float d3x = 3.0f*dux;
	const float d3y = 3.0f*duy;
	const float d3z = 3.0f*duz;
	const float s3x = 3.0f*sx;
	const float s3y = 3.0f*sy;
	const float s3z = 3.0f*sz;
	const float dxy = d3x+d3y;
	const float sxy = s3x+s3y;
	const float dxz = d3x+d3z;
	const float sxz = s3x+s3z;
	const float dyz = d3y+d3z;
	const float syz = s3y+s3z;
	const float dxmy = d3x-d3y;
	const float sxmy = s3x-s3y;
	const float dxmz = d3x-d3z;
	const float sxmz = s3x-s3z;
	const float dymz = d3y-d3z;
	const float symz = s3y-s3z;
	const float d0 = def_w0*rhon*h3;
	const float d1 = wr_s*fma(d3x, fma(0.5f, s3x, 1.0f), h3);
	const float d2 = wr_s*fma(-d3x, fma(-0.5f, s3x, 1.0f), h3);
	const float d3 = wr_s*fma(d3y, fma(0.5f, s3y, 1.0f), h3);
	const float d4 = wr_s*fma(-d3y, fma(-0.5f, s3y, 1.0f), h3);
	const float d5 = wr_s*fma(d3z, fma(0.5f, s3z, 1.0f), h3);
	const float d6 = wr_s*fma(-d3z, fma(-0.5f, s3z, 1.0f), h3);
	const float d7 = wr_e*fma(dxy, fma(0.5f, sxy, 1.0f), h3);
	const float d8 = wr_e*fma(-dxy, fma(-0.5f, sxy, 1.0f), h3);
	const float d9 = wr_e*fma(dxz, fma(0.5f, sxz, 1.0f), h3);
	const float d10 = wr_e*fma(-dxz, fma(-0.5f, sxz, 1.0f), h3);
	const float d11 = wr_e*fma(dyz, fma(0.5f, syz, 1.0f), h3);
	const float d12 = wr_e*fma(-dyz, fma(-0.5f, syz, 1.0f), h3);
	const float d13 = wr_e*fma(dxmy, fma(0.5f, sxmy, 1.0f), h3);
	const float d14 = wr_e*fma(-dxmy, fma(-0.5f, sxmy, 1.0f), h3);
	const float d15 = wr_e*fma(dxmz, fma(0.5f, sxmz, 1.0f), h3);
	const float d16 = wr_e*fma(-dxmz, fma(-0.5f, sxmz, 1.0f), h3);
	const float d17 = wr_e*fma(dymz, fma(0.5f, symz, 1.0f), h3);
	const float d18 = wr_e*fma(-dymz, fma(-0.5f, symz, 1.0f), h3);
	fhn[0] += d0;
	fhn[1] += d1;
	fhn[2] += d2;
	fhn[3] += d3;
	fhn[4] += d4;
	fhn[5] += d5;
	fhn[6] += d6;
	fhn[7] += d7;
	fhn[8] += d8;
	fhn[9] += d9;
	fhn[10] += d10;
	fhn[11] += d11;
	fhn[12] += d12;
	fhn[13] += d13;
	fhn[14] += d14;
	fhn[15] += d15;
	fhn[16] += d16;
	fhn[17] += d17;
	fhn[18] += d18;
	// D3Q19-Reihenfolge wie im REK-Block: 7 (+,+,0), 9 (+,0,+), 11 (0,+,+), 13 (+,-,0), 15 (+,0,-), 17 (0,+,-).
	const float jx = d1-d2+d7-d8+d9-d10+d13-d14+d15-d16;
	const float jy = d3-d4+d7-d8+d11-d12-d13+d14+d17-d18;
	const float jz = d5-d6+d9-d10+d11-d12-d15+d16-d17+d18;
	const float m = ((d0+d1+d2)+(d3+d4+d5+d6))+((d7+d8+d9+d10)+(d11+d12+d13+d14))+((d15+d16)+(d17+d18));
	return (float4)(jx, jy, jz, m);
}
)+"#endif"+R( // FAC_R1Q_AN
// ★★ iMEM-Facettenpfad (FACETTEN-IMEM.md, an Asmuth et al. 2021 Gl. 20-28 verankert, Revision
// 2026-08-16): Slip-Geschwindigkeit u_s statt Diagonalpaar-Tausch. JEDER Link mit solidem
// Streaming-Ursprung traegt (linkweise, nicht paarweise); der Zusatzterm q_i = 6 w_i (c_i*u_s)
// ist dieselbe Termform wie apply_moving_boundaries (rho_wall=1). Register: fhn[i] haelt
// f_out_opp(t-2) -- der Esoteric-Pull-BB ist ein ZWEI-Schritt-Umlauf (Gegenpruefer, Auflage 1);
// alle Formeln sind zeitindexfrei. 2x2-System in der Tangentialebene (Quer-Ziel 0 = Modell),
// Degenerationskaskade fuer Einzellink-Zellen, Klemmen machen Ist!=Soll im Akkumulator sichtbar.
float3 apply_facette_imem)+"("+R(const uxx n, float* fhn, const uxx* j, const global uchar* flags,
                        const global float* fac_geo, const global uint* fac_idx,
                        global float* fac_tau_acc, global uint* fac_tau_cnt, global uint* hits, const ulong t
)+"#ifdef FACETTEN_EMA"+R(
                        , global float* fac_us // EMA-Zustand 3 float je Facette (1 Zelle = 1 Facette: racefrei)
)+"#endif"+R( // FACETTEN_EMA
)+"#ifdef FACETTEN_PEMA"+R(
                        , global float* fac_pu // PEMA-Zustand 6 float je Facette: P-quer (xyz) + u-quer (xyz)
)+"#endif"+R( // FACETTEN_PEMA
)+"#ifdef FACETTEN_DIAGZ"+R(
                        , global float* fac_diag // 19-float-Kettenprotokoll ([16] Selektor, [17] alpha, [18] dp_ds) der Diagnose-Facette
)+"#endif"+R( // FACETTEN_DIAGZ
)+"#ifdef FACETTEN_ELIBB"+R(
                        , const global uchar* fac_q // ★ B2: q je Link (18 uchar je Facette, B1)
)+"#endif"+R( // FACETTEN_ELIBB
)+"#ifdef FACETTEN_NACHBAR"+R(
)+"#ifdef FAC_REK"+R(
                        , global float* fac_nb // ★ 23.09. spaet, Pruefbefund H-N2: NUR unter FAC_REK nicht-const. Der Akkumulator schreibt nach def_nb_roff+3. Ohne das #ifdef aenderte der Wegfall des const den emittierten Kernel auch bei CFD_FAC_REK=0, und damit war der AUS-Arm nicht mehr quelltextidentisch -- genau das, was ich behauptet hatte.
)+"#else"+R(
                        , const global float* fac_nb
)+"#endif"+R(
)+"#endif"+R( // FACETTEN_NACHBAR
)+"#ifdef FACETTEN_KDIAG"+R(
                        , global float* fac_kd // ★ Klassen-Diagnostik: 16 float je Facette (u_t, tw, twe, |P1|, s1, phi1, Rueckfall, Besuche, ut_ab, yw_ab, tw_angewandt, besuche_angewandt, [12..15] 05.09. Druckrest A / |A| / Ziel B / Geometrie C, alle nur ueber angewandte Besuche), racefrei (1 Zelle = 1 Facette)
)+"#endif"+R( // FACETTEN_KDIAG
)+") {"+R(
	uxx fbi; if(!f_bbox(n, &fbi)) return (float3)(0.0f,0.0f,0.0f);
	const uint fid = fac_fid(fac_idx, fbi);
	if(fid==0xFFFFFFFFu) return (float3)(0.0f,0.0f,0.0f);
	const uxx b = 8ul*(uxx)fid;
	const float nx=fac_geo[b], ny=fac_geo[b+1ul], nz=fac_geo[b+2ul], yw=fac_geo[b+3ul], faca=fac_geo[b+4ul];
)+"#ifdef FACETTEN_MESSNUR"+R(
	// ★★ MESS-NUR (CFD_FAC_MESSNUR; Pruefbefund B-4 vom 02.09.): Ausstieg VOR der ELIBB-Blende --
	// vorher lief die Blende (modifiziert fhn, bucht fac_tau_acc an der Treppe) noch mit, und der
	// "reine BB"-Arm war an kipp26 in Wahrheit Blende-BB. Jetzt ist MESS-NUR exakt reines Bounce-Back,
	// auch mit ELIBB-Emission. Slot 7 (Wirkpfad-Soll) und Slot 38 (MESSNUR-Wirkpfad; Umzug von 23,
	// Pruefbefund B-3: 20-26 gehoeren boden_eq/einlass_eq/schale_blend ueber den diag-Alias) zaehlen hier.
	if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[7]); atomic_inc(&hits[75]); } // Slot 75 (B70)
	return (float3)(0.0f,0.0f,0.0f);
)+"#endif"+R( // FACETTEN_MESSNUR
)+"#ifdef FACETTEN_ELIBB"+R(
	// ★★ REIHENFOLGE-FIX (2026-08-25 abends, nach g7): Blende ZUERST, DANN tastet das Wandmodell
	// den REKONSTRUIERTEN Zustand ab. Vorher lief abtasten -> solve -> Blende -> Additivterm: der
	// Solve kannte die Blende nicht und injizierte auf veraenderten Populationen -- der CPU-Harness
	// OHNE iMEM war sauber, jeder GPU-Arm MIT iMEM brach (Kugel-Cd -3,3/-4,95/-7,1 quer durch alle
	// Operanden- und q-Quellen-Arme). Die Blende ist GEOMETRIE (Randbedingung), keine Korrektur.
	// kipp0: qb=127 ueberall -> Blende ist Identitaet -> bitgleich (Anker haelt konstruktiv).
	float3 elibb_dp = (float3)(0.0f, 0.0f, 0.0f);
	{
		float r0, u0x, u0y, u0z;
		calculate_rho_u(fhn, &r0, &u0x, &u0y, &u0z);
		elibb_dp = elibb_rekonstruiere(fhn, j, flags, fac_q, fid, r0, u0x, u0y, u0z, nx, ny, nz, fac_tau_acc, hits, t);
	}
)+"#ifdef FACETTEN_ELIBB_PUR"+R(
	return (float3)(0.0f,0.0f,0.0f); // ★ Pur-Arm (CFD_FAC_ELIBB=2): NUR die Geometrie-Blende, kein Wandmodell -- Isolationsmessung
)+"#endif"+R( // FACETTEN_ELIBB_PUR
)+"#endif"+R( // FACETTEN_ELIBB
)+R(	float rhon, uxn, uyn, uzn;
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // Zustand VOR der Korrektur (Hans Abtastpunkt; unter ELIBB: NACH der Geometrie-Blende)
	const float und = nx*uxn+ny*uyn+nz*uzn;
	const float utx=uxn-und*nx, uty=uyn-und*ny, utz=uzn-und*nz;
	float ut = sqrt(utx*utx+uty*uty+utz*utz);
	if(t%def_zaehl_takt==0ul) atomic_inc(&hits[7]); // Wirkpfad (Soll = fac_N * ceil(n/100), wie Paararm)
)+"#ifdef FAC_REK"+R(
	// ★★ 24.09., Pruefbefund H5-2: der Konstantenspiegel stand HINTER dem ut-Tor. Dort heisst
	// "335 = 0" zweierlei -- der Block steht nicht im uebersetzten Geraetecode ODER alle
	// Facettenbesuche an Zaehlschritten sind am ut-Tor ausgestiegen. Der Host behauptet nur die
	// erste Ursache und brach darauf ab. Jetzt davor: der Spiegel belegt ausschliesslich, dass der
	// uebersetzte Kernel den Block traegt. Bleibt in #ifdef FAC_REK, der AUS-Arm ist unberuehrt.
	if(t%def_zaehl_takt==0ul) hits[335] = 0x5245464Bu;
)+"#endif"+R( // FAC_REK
	if(ut<1e-6f) { if(t%def_zaehl_takt==0ul) atomic_inc(&hits[9]); return (float3)(0.0f,0.0f,0.0f); }
)+"#ifdef FAC_REK_R3"+R(
	// ★★ R3 (Entscheid REKONSTRUKTION-PLAN.md §12): wo die statische Marke sitzt, setzt die
	// Rekonstruktion und der Solve wird uebersprungen -- zwei Aktoren an derselben Zelle sind nicht
	// auswertbar. Diese fuenf Werte tragen den Zustand bis zum Gate (Rueckfall) und zur Buchung.
	// WARUM EIN EIGENER ARM (CFD_FAC_REK=2): damit Tor-Wirkung und eps-Wirkung trennbar bleiben.
	// ★★ 23.09. abends BERICHTIGT. Hier stand "das Gate ist AUCH bei eps = 0 eine Physikaenderung,
	// es schaltet den Solve an allen Rang-0-Facetten ab (am kipp26 ein Drittel)". GEMESSEN ist das
	// am kipp26 FALSCH: Slot 331 = 0 von 53 195 580, der Hash des Tor-Arms ist der Anker.
	// Am 8-mm-FAHRZEUG dagegen Slot 331 = 30 335 von 6 995 090 -- dort arbeitet das Tor sehr wohl.
	// Die Trennung lohnt also, aber aus dem umgekehrten Grund: sie zeigt, WO das Tor ueberhaupt
	// etwas tut. Dass es das am Fahrzeug tut, stellt die Zensus-Invariante "statischer Rang 0 ist
	// die OBERGRENZE des Laufzeitrangs" in Frage -- offener Punkt, Slot 373 misst dagegen.
	bool rek_gate = false;
	float rek_dux = 0.0f;
	float rek_duy = 0.0f;
	float rek_duz = 0.0f;
	float rek_rho = 0.0f;
	float rek_g11 = 0.0f;
	float rek_sx = 0.0f;
	float rek_sy = 0.0f;
	float rek_sz = 0.0f;
	float rek_h3 = 0.0f;
)+"#endif"+R( // FAC_REK_R3
)+"#ifdef FAC_REK"+R(
	{
		const float rek_marke = fac_geo[b+7ul];
)+"#ifdef FAC_REK_S2"+R(
		// ★★ STUFE S2 (24.09.2026): die Amplitude ist KEIN Handwert mehr. Sie steht in
		// fac_nb[...roff+4] und wurde im VORSCHRITT aus dem Wandmodellziel bestimmt:
		//   rho*du = R1*t1  mit  R1 = -def_fac_tau*twe - P1
		// Die Gleichung ist erzwungen, nicht gewaehlt: nach der Doppelterm-Korrektur ist
		// fw.t1 = -P1_vor - rho*delta, und das soll def_fac_tau*twe sein.
		// LAG 1 IST BEWUSST UND MUSS GEMESSEN WERDEN. Der Wert stammt aus dem Vorschritt, weil
		// P1 erst nach der Momentenschleife feststeht, die Injektion aber davor laeuft. Die
		// Ein-Zellen-Linearisierung des Planungsschritts gibt einen Kontraktionsfaktor
		// G11roh/rho <= 1/3 -- das ist eine ABSCHAETZUNG, kein Stabilitaetsbeweis, und die
		// Periode-2-Mode an Wandzellen ist im Plan als real gefuehrt. Slot 393 misst die
		// Schrittaenderung, damit Konvergenz nicht behauptet, sondern gesehen wird.
		const float rek_eps = fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+4ul];
)+"#else"+R(
		const float rek_eps = fac_geo[b+6ul];
)+"#endif"+R( // FAC_REK_S2
		if(rek_marke>0.5f) {
			const bool rek_probe = (t%def_zaehl_takt==0ul);
			const float mxy_vor = rek_probe ? fhn[7]+fhn[8]-fhn[13]-fhn[14] : 0.0f;
			float rho_roh_vor = 0.0f;
			if(rek_probe) {
				rho_roh_vor = fhn[0];
				for(uint i=1u; i<def_velocity_set; i++) rho_roh_vor += fhn[i];
				rho_roh_vor += 1.0f;
				const float t1x_m = utx/ut;
				const uint kx = t1x_m<-0.5f ? 0u : (t1x_m<-0.1f ? 1u : (t1x_m<0.0f ? 2u : (t1x_m<0.1f ? 3u : (t1x_m<0.5f ? 4u : (t1x_m<0.9f ? 5u : (t1x_m<0.99f ? 6u : 7u))))));
				atomic_inc(&hits[336ul+(ulong)kx]);
				const uint kr = rhon<0.55f ? 0u : (rhon<0.7f ? 1u : (rhon<0.85f ? 2u : (rhon<0.95f ? 3u : (rhon<1.05f ? 4u : (rhon<1.2f ? 5u : (rhon<1.45f ? 6u : 7u))))));
				atomic_inc(&hits[344ul+(ulong)kr]);
)+"#ifdef FACETTEN_NACHBAR"+R(
				const float nb_ut = fac_nb[def_nb_stride*(ulong)fid];
				if(nb_ut>1e-6f) {
					const float tnbx = fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+0ul];
					const float tnby = fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+1ul];
					const float tnbz = fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+2ul];
					const uint kn = tnbx<-0.5f ? 0u : (tnbx<-0.1f ? 1u : (tnbx<0.0f ? 2u : (tnbx<0.1f ? 3u : (tnbx<0.5f ? 4u : (tnbx<0.9f ? 5u : (tnbx<0.99f ? 6u : 7u))))));
					atomic_inc(&hits[352ul+(ulong)kn]);
					const float cosw = (utx*tnbx+uty*tnby+utz*tnbz)/ut;
					const uint kc = cosw<-0.5f ? 0u : (cosw<0.0f ? 1u : (cosw<0.3f ? 2u : (cosw<0.6f ? 3u : (cosw<0.8f ? 4u : (cosw<0.95f ? 5u : (cosw<0.999f ? 6u : 7u))))));
					atomic_inc(&hits[360ul+(ulong)kc]);
					const float nlen = sqrt(tnbx*tnbx+tnby*tnby+tnbz*tnbz);
					const float ntan = tnbx*nx+tnby*ny+tnbz*nz;
					const bool norm_ok = (fabs(nlen-1.0f)<=1e-5f);
					const bool tang_ok = (fabs(ntan)<=1e-3f);
					if(!norm_ok||!tang_ok) atomic_inc(&hits[369]);
				}
				else atomic_inc(&hits[368]);
)+"#endif"+R(
			}
			if(rek_probe) atomic_inc(&hits[328]);
			const float rek_inv = 1.0f/ut;
			const float dux = rek_eps*utx*rek_inv;
			const float duy = rek_eps*uty*rek_inv;
			const float duz = rek_eps*utz*rek_inv;
)+"#ifdef FACETTEN_NACHBAR"+R(
			// ★★ IMPULS-AKKUMULATOR (23.09.2026, Stufe A2). Summiert den von der Rekonstruktion je Schritt
			// EINGETRAGENEN x-Impuls rho*du_x, ungegatet (jeder Schritt, nicht nur Zaehlschritte) -- die
			// Kraftbilanz K2 mittelt ebenfalls ueber JEDEN Schritt des Fensters, ein Stichprobenzaehler
			// waere nicht vergleichbar. rho ist das GEKLEMMTE rhon, dasselbe, das oben wr_s/wr_e bildet:
			// an 85 % der markierten Besuche steht es auf der Klemme, ein roh nachgerechnetes rho waere
			// bis 43 % zu gross. du_x ist das TATSAECHLICH angewandte, nicht das beabsichtigte.
			// Analytisch ist Summe_i c_i Df_i == rho*du EXAKT (das dritte Gittermoment von D3Q19
			// verschwindet identisch, der quadratische Term traegt zum ersten Moment nichts bei).
			// Racefrei ohne Atomik, weil 1 Zelle = 1 Facette gilt und das in lbm.cpp bewacht ist --
			// derselbe Grund, aus dem fac_tau_acc nicht-atomar akkumulieren darf.
			fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+3ul] += rhon*dux;
)+"#endif"+R( // FACETTEN_NACHBAR
)+"#ifdef FAC_REK_R3"+R(
			rek_dux = dux;
			rek_duy = duy;
			rek_duz = duz;
			rek_rho = rhon;
)+"#endif"+R( // FAC_REK_R3
			const float sx = fma(2.0f, uxn, dux);
			const float sy = fma(2.0f, uyn, duy);
			const float sz = fma(2.0f, uzn, duz);
			const float dc3 = -3.0f*fma(sx, dux, fma(sy, duy, sz*duz));
			const float wr_s = def_ws*rhon;
			const float wr_e = def_we*rhon;
			const float d3x = 3.0f*dux;
			const float d3y = 3.0f*duy;
			const float d3z = 3.0f*duz;
			const float s3x = 3.0f*sx;
			const float s3y = 3.0f*sy;
			const float s3z = 3.0f*sz;
			const float h3 = 0.5f*dc3;
)+"#ifdef FAC_REK_R3"+R(
			rek_sx = sx;
			rek_sy = sy;
			rek_sz = sz;
			rek_h3 = h3;
			// rek_gate wird ERST HIER gesetzt, nicht oben bei rek_dux. Pruefbefund N1 (24.09.):
			// die ACHT Injektionsgroessen werden an zwei Stellen mit einem Rechenblock dazwischen
			// gefuellt. Ein kuenftiger Frueh-Ausstieg dazwischen liesse rek_gate wahr und
			// rek_s*/rek_h3 auf 0 stehen -- rek_df waere still falsch, ohne dass ein Zaehler feuert.
			// Invariante: rek_gate == true heisst, alle ACHT sind gesetzt.
			// ★ AUSDRUECKLICH AUSGENOMMEN (Pruefbefund M6, 24.09.): rek_g11. Es entsteht erst mit
			// G11roh in der Momentenschleife, also HINTER dieser Zeile, und wird unter rek_gate
			// fuer das Histogramm 373..377 gelesen. Ein Frueh-Ausstieg zwischen hier und dort
			// liefert rek_gate == true bei rek_g11 == 0, und alle Marken landen still im untersten
			// Fach. Diese Invariante deckt rek_g11 NICHT ab; wer dort etwas einfuegt, prueft selbst.
			rek_gate = true;
)+"#endif"+R( // FAC_REK_R3
			fhn[0] += def_w0*rhon*h3;
			fhn[1] += wr_s*fma(d3x, fma(0.5f, s3x, 1.0f), h3);
			fhn[2] += wr_s*fma(-d3x, fma(-0.5f, s3x, 1.0f), h3);
			fhn[3] += wr_s*fma(d3y, fma(0.5f, s3y, 1.0f), h3);
			fhn[4] += wr_s*fma(-d3y, fma(-0.5f, s3y, 1.0f), h3);
			fhn[5] += wr_s*fma(d3z, fma(0.5f, s3z, 1.0f), h3);
			fhn[6] += wr_s*fma(-d3z, fma(-0.5f, s3z, 1.0f), h3);
			const float dxy = d3x+d3y;
			const float sxy = s3x+s3y;
			const float dxz = d3x+d3z;
			const float sxz = s3x+s3z;
			const float dyz = d3y+d3z;
			const float syz = s3y+s3z;
			const float dxmy = d3x-d3y;
			const float sxmy = s3x-s3y;
			const float dxmz = d3x-d3z;
			const float sxmz = s3x-s3z;
			const float dymz = d3y-d3z;
			const float symz = s3y-s3z;
			fhn[7] += wr_e*fma(dxy, fma(0.5f, sxy, 1.0f), h3);
			fhn[8] += wr_e*fma(-dxy, fma(-0.5f, sxy, 1.0f), h3);
			fhn[9] += wr_e*fma(dxz, fma(0.5f, sxz, 1.0f), h3);
			fhn[10] += wr_e*fma(-dxz, fma(-0.5f, sxz, 1.0f), h3);
			fhn[11] += wr_e*fma(dyz, fma(0.5f, syz, 1.0f), h3);
			fhn[12] += wr_e*fma(-dyz, fma(-0.5f, syz, 1.0f), h3);
			fhn[13] += wr_e*fma(dxmy, fma(0.5f, sxmy, 1.0f), h3);
			fhn[14] += wr_e*fma(-dxmy, fma(-0.5f, sxmy, 1.0f), h3);
			fhn[15] += wr_e*fma(dxmz, fma(0.5f, sxmz, 1.0f), h3);
			fhn[16] += wr_e*fma(-dxmz, fma(-0.5f, sxmz, 1.0f), h3);
			fhn[17] += wr_e*fma(dymz, fma(0.5f, symz, 1.0f), h3);
			fhn[18] += wr_e*fma(-dymz, fma(-0.5f, symz, 1.0f), h3);
			if(rek_probe) {
				float rr_roh = fhn[0];
				for(uint i=1u; i<def_velocity_set; i++) rr_roh += fhn[i];
				rr_roh += 1.0f;
				float rr, rux, ruy, ruz;
				calculate_rho_u(fhn, &rr, &rux, &ruy, &ruz);
				const float wx = rux-uxn;
				const float wy = ruy-uyn;
				const float wz = ruz-uzn;
				const float wbetrag = sqrt(wx*wx+wy*wy+wz*wz);
				const float ubetrag = sqrt(uxn*uxn+uyn*uyn+uzn*uzn);
				if(wbetrag>1e-6f*fmax(ubetrag, 1e-12f)) atomic_inc(&hits[329]);
				const float soll_x = uxn+dux;
				const float soll_y = uyn+duy;
				const float soll_z = uzn+duz;
				const float rx = rux-soll_x;
				const float ry = ruy-soll_y;
				const float rz = ruz-soll_z;
				const float rbetrag = sqrt(rx*rx+ry*ry+rz*rz);
				const float rtol = fma(1.0E-3f, fabs(rek_eps), fmax(1.0E-6f*fmax(ubetrag, 1e-12f), 5.0E-8f));
				if(rbetrag>rtol) atomic_inc(&hits[330]);
				const float dm = fabs(rr_roh-rho_roh_vor);
				if(dm>5.0E-7f*rho_roh_vor) atomic_inc(&hits[332]);
				const float mxy_nach = fhn[7]+fhn[8]-fhn[13]-fhn[14];
				const float mxy_soll = rhon*(fma(uxn, duy, dux*uyn)+dux*duy);
				const float mxy_rest = fabs((mxy_nach-mxy_vor)-mxy_soll);
				const float mxy_tol = fma(1.0E-3f, fabs(mxy_soll), 1.0E-8f);
				if(mxy_rest>mxy_tol) atomic_inc(&hits[333]);
				if(fabs(mxy_soll)<=1.0E-7f) atomic_inc(&hits[334]);
			}
		}
	}
)+"#endif"+R(
 // Slot 9: iMEM modifiziert bei ut~0 GAR NICHT (t-Basis undefiniert; dokumentierte Abweichung vom Paararm, der den Tausch trotzdem macht)
	// ★★ NACHBARABTASTUNG (CFD_FAC_NACHBAR, 30.08.2026, Weg-1 Stufe 3). BEFUND, der sie ausloest
	// (Klassen-Diagnostik CFD_FAC_KDIAG am 26-Grad-Kanal, V3b-Konfiguration): die konkave Eckzelle
	// (8 eigene Solid-Links, y_w 0,18) tastet u_t = 0,0051 ab, die freie Zelle ueber derselben
	// Stufe 0,0224 -- Faktor 4,4. Die Eckzelle liegt im STUFENSCHATTEN; ihr u_t ist kein
	// Grenzschichtwert, sondern der Rest einer abgeschatteten Rezirkulation. Spalding macht daraus
	// tau ~ u^2, also Ziel/Ist 0,04. Abhilfe: den EINGANG (u_t und Wandabstand) aus der zweiten
	// Fluidzelle ENTLANG DER NORMALE nehmen -- dort steht das Profil frei. Wie bei UTKORR wirkt das
	// NUR auf Y/u+/u_tau; Tangentialbasis, Klemme tw_max und die Gates bleiben auf dem lokalen ut,
	// damit Budget und Stabilitaet unveraendert bleiben. ★ 03.09.: der Wert kommt aus dem Kernel
	// fac_nachbar_ab des VORSCHRITTS (fertiges u-Feld, eigener Launch nach stream_collide). Der fruehere
	// Direktzugriff u[nb] im selben Kernel war gemessen NICHT bitreproduzierbar (xu_det_mit_a/b: cf
	// 0,00073682648 gegen 0,00073630592 bei identischer Konfiguration; ohne NACHBAR bitgleich).
	float ut_ab = ut, yw_ab = yw; // Default: eigene Zelle = bisheriges Verhalten
	bool nachbar_ab = false; // ★ 03.09. (Planungsagent-Auflage): true NUR bei angewandter Nachbarabtastung -- UTKORR ist die 3/2-BB-Deflation der EIGENEN Zelle (P1 ~ -u/3) und darf den Nachbarwert nicht skalieren; ohne FACETTEN_NACHBAR konstant false (bitgleich)
)+"#ifdef FACETTEN_NACHBAR"+R(
	{ // ★ 03.09. deterministisch: Werte aus dem Kernel fac_nachbar_ab des Vorschritts -- kein u-Zugriff im selben Launch mehr
		const float utb = fac_nb[def_nb_stride*(ulong)fid]; // def_nb_stride: 2, unter FACETTEN_APG 5 (grad rho in [2..4])
		if(utb>1e-6f) {
			ut_ab = utb; nachbar_ab = true; yw_ab = fac_nb[def_nb_stride*(ulong)fid+1ul]; // Wandabstand der Abtastzelle
			// ★ 03.09. SAETTIGUNG wie Slot 76: am 4-mm-Fahrzeug laeuft dieser Zaehler auf 3,13 M Facetten x ~501
			// Stichproben = 1,57e9 = 37 % des uint-Bereichs (Luft nur Faktor 2,7). Ohne Schutz wickelte er bei
			// laengerem T_END oder feinerem Gitter STILL und der Report meldete eine falsche Prozentzahl.
			if(t%def_zaehl_takt==0ul&&hits[72]<0xF0000000u) atomic_inc(&hits[72]); // Slot 72: Nachbarabtastung angewandt (saettigend)
		} else if(utb<0.0f) { if(t%def_zaehl_takt==0ul&&hits[73]<0xF0000000u) atomic_inc(&hits[73]); } // Slot 73: kein Fluidnachbar in Normalenrichtung -- eigene Zelle (saettigend)
		else if(t%def_zaehl_takt==0ul&&hits[74]<0xF0000000u) atomic_inc(&hits[74]); // Slot 74: Nachbar gefunden, steht aber still (saettigend)
	}
)+"#endif"+R( // FACETTEN_NACHBAR
	float tw=0.0f, twe=0.0f; // Spalding-Kette WOERTLICH wie Paararm (Slots 8 seit R3 gegatet); unter PEMA wird twe unten aus dem gefilterten u ueberschrieben
	{
		// ★★ 3/2-ABTASTPUNKT-MESSARM (2026-08-25, CFD_FAC_UTKORR, Default 1,0 = bitgleich).
		// GEMESSEN (Ebene-Wand-Agent, drei Aufloesungen): das hier abgetastete u_t betraegt
		// 0,698-0,707 des wahren u der ersten Lage -- die BB-Theorie sagt exakt 2/3 (die
		// Wandlinks tragen zum Abtastzeitpunkt noch die No-Slip-Reflexion, P1 ~ -u/3).
		// Spalding macht daraus tau ~ u^2 -> Faktor ~2 zu wenig Wandschubspannung; das ist
		// der Massstab der zentralen c_f-Luecke (0,00154 gegen Referenz 0,00344). Der Faktor
		// wirkt NUR auf den Wandmodell-EINGANG (Y, u+, u_tau); die Tangentialbasis und die
		// Stabilitaetsklemme tw_max laufen weiter auf dem rohen ut. Deklarierter Messarm,
		// NICHT still eingebaut -- Soll-Faktor der Theorie: 3/2.
		const float ut_wm = nachbar_ab ? ut_ab : ut_ab*def_fac_utkorr; // ★ 03.09.: UTKORR nur an Eigenabtastung (Slot 73/74); am Nachbarn (Slot 72) gilt die BB-Deflation nicht. Ohne FACETTEN_NACHBAR: ut_ab*def_fac_utkorr wie bisher (bitgleich)
		const float Y  = ut_wm*((2.0f*yw_ab)*def_fac_Y);
		const float up = wf_spalding_uplus(Y);
		const float utau = ut_wm/up;
		tw = rhon*utau*utau;
		const float tw_max = 0.5f*rhon*ut;
)+"#ifndef FACETTEN_PEMA"+R(
)+"#ifndef FACETTEN_APG"+R(
		// Auditor A Befund 3 (07.09.): GENAU EINMAL JE BESUCH. Vorher standen zwei Zaehlstellen --
		// eine in der Klemme, eine hinter twf = tw*faca. Feuert die erste, ist tw == tw_max, also
		// twf = tw_max*faca, und an jeder NICHT achsparallelen Facette ist faca > 1: die zweite
		// feuerte zwingend mit. Slot 8 lief bis zum Doppelten der Besuchszahl und war als
		// tau-Klemme N (Host, ohne Nenner) nicht lesbar. Am 4 mm auch eine Kopfhoehenfrage: Slot 8
		// saettigt nicht und stand im schlimmsten Fall bei 79 Prozent des uint-Bereichs. Die
		// PEMA/APG-Zweige wurden im Tiefen-Audit R2 genau dafuer saniert; der Basisarm nicht.
		if((tw>tw_max||tw*faca>tw_max)&&t%def_zaehl_takt==0ul) atomic_inc(&hits[8]);
)+"#endif"+R( // FACETTEN_APG
)+"#endif"+R( // FACETTEN_PEMA
		if(tw>tw_max) { tw = tw_max; }
		const float twf = tw*faca;
)+"#ifndef FACETTEN_PEMA"+R(
)+"#ifndef FACETTEN_APG"+R(
		// (die zweite Zaehlstelle stand hier -- Befund 3, jetzt oberhalb zusammengefasst)
)+"#endif"+R( // FACETTEN_APG
)+"#endif"+R( // FACETTEN_PEMA
		twe = fmin(twf, tw_max);
	}
	float t1x=utx/ut, t1y=uty/ut, t1z=utz/ut;                     // Tangentialbasis (Gl. 6; unter PEMA aus dem gefilterten u neu gesetzt)
	float t2x=ny*t1z-nz*t1y, t2y=nz*t1x-nx*t1z, t2z=nx*t1y-ny*t1x;
)+"#ifdef FACETTEN_APG"+R(
	float fac_dpds=0.0f;
	{ // ★ APG, UMGEBAUT 16.09.2026 (PLAN-APG-2026-09-16.md §A/§B). Duennschicht-Impulsbilanz tau(y) = tau_w + y*dp/ds, aufgeloest an
		// der ABTASTHOEHE y_ab -- Befund A1: Spalding tastet seit NACHBAR (03.09.) an y_ab ab, die Korrektur nahm y_w der eigenen
		// Zelle (bis Faktor 3 zu klein); ohne NACHBAR ist yw_ab == yw (bitgleich zum alten Zweig). dp/ds = (grad rho . t1)/3 mit
		// grad rho aus dem Vorkernel fac_nachbar_ab (deterministisch, aus den DDFs; Befunde A2/A3). KEIN rho-Zugriff mehr in
		// diesem Kernel. kappa = def_fac_apg: herleitbar ist 1, 0,5 ist ein deklarierter Interim (Befund A7).
		const ulong nbb = def_nb_stride*(ulong)fid;
		const float gx=fac_nb[nbb+2ul], gy=fac_nb[nbb+3ul], gz=fac_nb[nbb+4ul];
		fac_dpds = (gx*t1x+gy*t1y+gz*t1z)*(1.0f/3.0f); // p = rho*c_s^2 = rho/3
		const float korr = def_fac_apg*yw_ab*fac_dpds;
		const bool zt = (t%def_zaehl_takt==0ul);
		if(zt) { // Diagnostik (Iron Rule 3): Zwischenergebnisse zaehlbar, nicht nur die Endkraft
			if(hits[308]<0xF0000000u) atomic_inc(&hits[308]);                       // APG-Zweig besucht, Soll = [7]-[9]
			if(fac_dpds>0.0f) { if(hits[311]<0xF0000000u) atomic_inc(&hits[311]); } // dp/ds > 0: Gegendruck (APG)
			else if(fac_dpds<0.0f) { if(hits[312]<0xF0000000u) atomic_inc(&hits[312]); } // dp/ds < 0: FPG
			if(tw>0.0f) { const float r_ = fabs(korr)/tw; const uint hb_ = r_<0.1f ? 313u : (r_<0.5f ? 314u : (r_<1.0f ? 315u : 316u)); // Autoritaet |kappa*y*dp/ds|/tw
				if(hits[hb_]<0xF0000000u) atomic_inc(&hits[hb_]); }
		}
)+"#ifdef FACETTEN_APG_MOZ"+R(
		// ★★ 22.09.2026 MOZAFFARI-JACOB-SAGAUT 2024 (Flow Turbulence Combust 106:3) statt der linearen
		// Duennschichtkorrektur. ANLASS, gemessen am 22.09. (Tagesprotokoll B15): die lineare Form hat
		// Autoritaet |kappa*y_ab*dp/ds|/tau_w >= 1 in 97,1 % der Besuche, und 97,06 % werden geklemmt --
		// sie ist ein Vorzeichendetektor, keine Korrektur.
		// WARUM DIESE FORM TRAEGT, und es ist NICHT die Saettigung: mit A = |korr|/tw gilt exakt
		//   alpha_p = nu*(dp/ds)/(rho*u_tau^3) = A*nu/(y_ab*u_tau) = A / y+_ab.
		// Die gemessene Autoritaet wird also durch y+_ab geteilt (Facetten-y+ Median 148 an y_w, an y_ab
		// ~150-450). Die 97,1 % mit A >= 1 landen damit im EMPFINDLICHEN Band von f, nicht in der Saettigung.
		// OB das so ist, ist bis heute UNBELEGT -- das oberste Autoritaetsfach [316] ist nach oben offen.
		// Genau deshalb sind die Faecher 318..325 unten kein Beiwerk, sondern der Befund.
		// ALGEBRAISCH UMGESTELLT: f = 1 - C/(1 + a0/ap) statt 1 - C*ap/(ap+a0). Gleiche Funktion, aber
		// total auf (0, inf]: kein inf/inf bei ap -> inf (u_tau -> 0), keine Ausloeschung im Nenner.
		// FPG-AST DEKLARIERT f == 1: die Paperform hat bei ap = -a0 = -0,005 eine POLSTELLE, und mit
		// 18,5 Mio FPG-Besuchen (gemessen) ist ihr Umfeld sicher besetzt -- in float32 waere |f| dort bis
		// 4,3e6. Physikalisch ist der Fit an APG kalibriert (NACA-4412, Ahmed); unter FPG unterschaetzt
		// das Loggesetz tau_w, eine Daempfung haette dort das falsche Vorzeichen. Die symmetrische
		// Fortsetzung waere polfrei, saettigt im FPG-Ast aber bei f^2 = 1,95 -- also fast exakt auf der
		// Obenklemme 2*tw, die wir gerade als Artefakt verabschieden. Deshalb NICHT als Vorgabe.
		// KEINE Klemme mehr: f liegt in [1-C, 1] = [0,6; 1], also tw/tw_Spalding in [0,36; 1] (Waechter 0 < C <= 1 in lbm.cpp).
		// Die Endklemme unten (Slot 8, tw*faca > 0,5*rhon*ut) bleibt -- sie feuert unter MOZ nur bei faca > 1, seltener als linear (2A-N1).
		// Die alte Klemme KANN konstruktiv nicht mehr feuern; ihr Nachweis laeuft ueber die
		// Schattenzaehler 326/327, die dieselbe Bedingung zaehlen, ohne die Physik anzufassen.
		// ★ BEZUG (Pruefbefund A-1/B-M7, 22.09.): tw ist hier das GEKLEMMTE Spalding-tw (Klemme tw_max steht oben im
		// Spalding-Block, VOR diesem Zweig). Die erste Fassung baute tw aus dem UNGEKLEMMTEN u_tau neu auf und hob damit
		// die Klemme fuer den MOZ-Arm still auf -- zwei Aenderungen in einem Arm. Jetzt: u_tau = sqrt(tw/rho) aus dem
		// geklemmten tw, und tw_neu = tw*f^2 -- auf dem FPG-Ast (f == 1) damit BITGLEICH zum Spalding-Wert.
		{	const float nu_mol = 0.5f/def_fac_Y; // ★ NICHT def_fac_nu: das wird nur unter FACETTEN_UW emittiert (lbm.cpp), in der Standardzeile gibt es das Makro GAR NICHT -- JIT-Fehler statt falscher Zahl.
			const float tw_sp_ = tw; // geklemmtes Spalding-tw: Bezug von f, der Schattenzaehler [326]/[327] und (oben) der Autoritaet [313..316]
			const float utau_c = sqrt(tw_sp_/rhon); // u_tau aus dem geklemmten tw
			float fm = 1.0f; uint fb_ = 318u;
			const float u3 = utau_c*utau_c*utau_c;
			const float ap = (u3>0.0f) ? nu_mol*fac_dpds/(rhon*u3) : 0.0f;
			if(u3<=0.0f||(as_uint(ap)&0x7F800000u)==0x7F800000u) { fm = 1.0f; fb_ = 317u; } // Entartung, Soll 0 -- NaN/Inf-Bit-Test statt isfinite (unter -cl-finite-math-only toter Code, Gross-Audit M / Pruefbefund A-3)
			else if(ap>0.0f) {
				fm = 1.0f - def_fac_apg_c/(1.0f + def_fac_apg_ap0/ap);
				fb_ = fm>=0.99f ? 318u : (fm>=0.95f ? 319u : (fm>=0.90f ? 320u : (fm>=0.80f ? 321u
				    : (fm>=0.70f ? 322u : (fm>=0.65f ? 323u : (fm>=0.62f ? 324u : 325u))))));
			}
			tw = tw_sp_*(fm*fm);
			if(zt) {
				if(hits[fb_]<0xF0000000u) atomic_inc(&hits[fb_]);
				// Schatten der ENTFALLENEN Klemme: exakt die Bedingung, die 309/310 zaehlten.
				if(tw_sp_>0.0f&&korr>tw_sp_)  { if(hits[326]<0xF0000000u) atomic_inc(&hits[326]); } // haette unten geklemmt
				if(tw_sp_>0.0f&&-korr>tw_sp_) { if(hits[327]<0xF0000000u) atomic_inc(&hits[327]); } // haette oben geklemmt
			}
		}
)+"#else"+R(
		float tw1 = tw-korr;
		// RELATIVE Kappung [0, 2*tw] (Lauf-4-Befund: 46 % 0-Klemmen). Slot 19 = beide Klemmen zusammen (alt, Bericht), 309/310 getrennt (neu).
		if(tw1<0.0f) { tw1=0.0f; if(zt) { atomic_inc(&hits[19]); if(hits[309]<0xF0000000u) atomic_inc(&hits[309]); } }
		else if(tw1>2.0f*tw) { tw1=2.0f*tw; if(zt) { atomic_inc(&hits[19]); if(hits[310]<0xF0000000u) atomic_inc(&hits[310]); } }
		tw = tw1;
)+"#endif"+R(
		if(tw*faca>0.5f*rhon*ut&&zt) atomic_inc(&hits[8]); // Tiefen-Audit A1-B2: Klemme der NACH-APG-Kette zaehlen (Kopf zaehlte die verworfene Vor-APG-Kette)
		twe = fmin(tw*faca, 0.5f*rhon*ut);
	}
)+"#endif"+R( // FACETTEN_APG
	float G11=0.0f, G22=0.0f, G12=0.0f, P1=0.0f, P2=0.0f;         // Linkmengen-Momente (Gl. 4/7)
)+"#ifdef FAC_REK_R3"+R(
	float rek_dp1=0.0f;
	float rek_dp2=0.0f;
	float rek_dpn=0.0f;
)+"#endif"+R( // FAC_REK_R3
)+"#ifdef FACETTEN_PEMA"+R(
	float Pvx=0.0f, Pvy=0.0f, Pvz=0.0f;
)+"#endif"+R( // FACETTEN_PEMA
	float S1x=0.0f, S1y=0.0f, S1z=0.0f, Sn1=0.0f, Sn2=0.0f, Snn=0.0f; // S1 KOMPONENTENWEISE (Auflage 2); Snn fuer die 3x3
)+"#ifdef FACETTEN_ALPHA"+R(
	float S0=0.0f; // J4-alpha: Summe w_i ueber die Wandlinks (>= w(1), sobald ein Link existiert)
)+"#endif"+R( // FACETTEN_ALPHA
	for(uint i=1u; i<def_velocity_set; i++) { // compiler-entrollt, Muster apply_moving_boundaries
		const uint ib = (i%2u==1u) ? i+1u : i-1u; // Streaming-Ursprung von fhn[i] ist j[opposite(i)]
		if((flags[j[ib]]&TYPE_BO)!=TYPE_S) continue; // linkweises Gate (schliesst TYPE_E/TYPE_MS aus)
		const float cx=c(i), cy=c(def_velocity_set+i), cz=c(2u*def_velocity_set+i);
		const float wi=w(i);
		const float ct1=cx*t1x+cy*t1y+cz*t1z, ct2=cx*t2x+cy*t2y+cz*t2z, cn=cx*nx+cy*ny+cz*nz;
		G11 = fma(6.0f*wi, ct1*ct1, G11); G22 = fma(6.0f*wi, ct2*ct2, G22); G12 = fma(6.0f*wi, ct1*ct2, G12);
		P1 = fma(2.0f*ct1, fhn[i], P1); P2 = fma(2.0f*ct2, fhn[i], P2); // Phi^f-Tangentialkomponenten (geshiftete DDFs; Offset hebt sich zellweise NUR bei symmetrischer Linkmenge -- an Treppen ist P1 der DEVIATORISCHE Austausch, im Flaechenintegral exakt 0: WM-Blick B)
)+"#ifdef FACETTEN_PEMA"+R(
		Pvx = fma(2.0f*cx, fhn[i], Pvx); Pvy = fma(2.0f*cy, fhn[i], Pvy); Pvz = fma(2.0f*cz, fhn[i], Pvz); // Phi^f als xyz-Vektor (Filterrahmen)
)+"#endif"+R( // FACETTEN_PEMA
)+"#ifdef FAC_REK_R3"+R(
		// DOPPELTERM (24.09.2026): fhn traegt an den Marken schon das Df der Rekonstruktion, P1/P2
		// oben zaehlen es also mit -- und unter R3 ist phi = P, die Wandkraft fuehrt es ein ZWEITES
		// Mal. Hier wird Df je Wandlink EXAKT nachgebildet (dieselbe Faktorisierung wie die
		// Injektion, siehe fhn[1] dort: wr_s*fma(d3x, fma(0.5f, s3x, 1.0f), h3)) und auf t1/t2
		// projiziert. EXAKT heisst: kein Ordnungsargument, kein vernachlaessigter Term -- der
		// quadratische Anteil und der -1,5(du.s)-Anteil sind mit drin. Bei unmarkierter Zelle sind
		// rek_rho und alle rek_du* null, also Df == 0 und die Summen bleiben null.
		const float rek_cdu = fma(cx, rek_dux, fma(cy, rek_duy, cz*rek_duz));
		const float rek_cs = fma(cx, rek_sx, fma(cy, rek_sy, cz*rek_sz));
		const float rek_df = wi*rek_rho*fma(3.0f*rek_cdu, fma(1.5f, rek_cs, 1.0f), rek_h3);
		rek_dp1 = fma(2.0f*ct1, rek_df, rek_dp1);
		rek_dp2 = fma(2.0f*ct2, rek_df, rek_dp2);
		// NORMALANTEIL des Wandlink-Flusses. NICHT dasselbe wie [372]: der prueft |du.n|, und du
		// steht konstruktiv tangential, also ist [372] konstruktiv 0. DP_n dagegen ist der
		// Normalanteil des FLUSSES ueber die Wandlinks, und der verschwindet nur an der EBENEN
		// Wand. An einer Einzellink-Zelle -- und 92 % der Rang-0-Facetten haben einen Link --
		// ist c.n von null verschieden, also auch DP_n. Planungsschritt S2, Befund H1 (24.09.).
		rek_dpn = fma(2.0f*cn, rek_df, rek_dpn);
)+"#endif"+R( // FAC_REK_R3
		S1x = fma(wi, cx, S1x); S1y = fma(wi, cy, S1y); S1z = fma(wi, cz, S1z);
		Sn1 = fma(6.0f*wi, ct1*cn, Sn1); Sn2 = fma(6.0f*wi, ct2*cn, Sn2);
		Snn = fma(6.0f*wi, cn*cn, Snn); // 3x3-Iteration: Normalautoritaet
)+"#ifdef FACETTEN_ALPHA"+R(
		S0 += wi;
)+"#endif"+R( // FACETTEN_ALPHA
	}
)+R(	// ★★ 2026-08-25, MESSUNG VOR DEM EINGRIFF. fhn traegt die STOERFORM f^ = f - w_i. Der wahre
	// tangentiale Impulsfluss durch die Wandlinks ist Sum 2 ct f_i = P1 + 2*(S1.t1); der Offset
	// 2*(S1.t1) faellt bei symmetrischer Linkmenge (ebene Wand) exakt weg, an der Treppe nicht.
	// Bevor irgendetwas daran geaendert wird, wird seine GROESSE gegen das Ziel def_fac_tau*twe
	// gemessen -- Dekadenhistogramm 49..53 fuer den Offset, 54..58 fuer P1 selbst.
	// ★ Pruefbefund 3-C: dieser Block steht UNBEDINGT (kein #ifdef). Die Loesung bleibt bitgleich --
	// die Zaehler koppeln nicht zurueck -- aber die Behauptung "neue Slots nur unter #ifdef-Emission"
	// in lbm.cpp gilt fuer ihn nicht. Hiermit richtiggestellt statt stillschweigend.
	{
		const float B1o=S1x*t1x+S1y*t1y+S1z*t1z, B2o=S1x*t2x+S1y*t2y+S1z*t2z;
		// ★ Pruefbefund 3-B (2026-08-25): fac_N ~ 1e6 mal 5000 Abtastungen sprengt uint. Hash-Stichprobe
		// ueber die Facetten-Nummer (jede 64.), damit die Bins nicht wickeln.
		if(t%def_zaehl_takt==0ul&&((fid*2654435761u)&4227858432u)==0u) {
			const float zi_ = def_fac_tau*twe; if(zi_>0.0f) { // ★ Pruefbefund B(b): bei Nullziel-Armen (def_fac_tau=0) waere jeder Wert im obersten Bin
			// ★ FIX 2026-08-27 (Planungsagent Schritt 2, Befund 2.1): hier stand "if(zi_<=0.0f) return;".
			// Das return verliess die GANZE Funktion, nicht nur diesen Histogrammblock -- in einem
			// Nullziel-Arm (def_fac_tau=0, setup.cpp:1646) wurden dadurch fuer jede 64. Facette auf
			// jedem 100. Schritt Solve, Pass 2 UND die Buchung uebersprungen, obwohl Slot 7 oben schon
			// gezaehlt hatte: eine stille Feldaenderung und ein Ist!=Soll in der Buchung, ausgerechnet
			// in dem Arm, der die Untergrenze des Ziel-Intervalls messen soll. Bei def_fac_tau>0 ist
			// der Zweig unveraendert (zi_>0 lief vorher durch und laeuft jetzt durch) -- alle
			// bestehenden Arme bleiben bitgleich, korrigiert wird ausschliesslich das Nullziel.
			const float ro_ = sqrt(fma(2.0f*B1o,2.0f*B1o,4.0f*B2o*B2o))/zi_;
			const float rp_ = sqrt(fma(P1,P1,P2*P2))/zi_;
			atomic_inc(&hits[49u+(ro_<0.01f?0u:(ro_<0.1f?1u:(ro_<1.0f?2u:(ro_<10.0f?3u:4u))))]);
			// ★ Pruefbefund B(b): die alten Grenzen 0,01/0,1/1/10 legten die EBENE Wand zu 100 % in den
			// offenen obersten Bin -- null Information, und die als "bimodal" gelesene 45-Grad-Verteilung
			// war ein Saettigungsartefakt. Grenzen dorthin, wo die Daten liegen.
			atomic_inc(&hits[54u+(rp_<1.0f?0u:(rp_<10.0f?1u:(rp_<100.0f?2u:(rp_<1000.0f?3u:4u))))]);
			// ★ 05.09.2026 HINWEIS: der Zaehler fuer den RESTANTEIL DES DRUCKTERMS (Slots 118..122) stand
			// zuerst HIER und wurde nach Pruefbefund H-4 in den Zielerfuellungsblock (~2585) VERLEGT, weil
			// dieser Block nicht auf pass2_an gegatet ist. Die Herleitung steht dort.
			} // zi_>0
		}
	}
	// ★ 2026-08-25 ZURUECKGENOMMEN: hier standen Rohkopien fuer einen "A2-Rueckfall", der die
	// Einzellink-Facetten aus dem Slot-13-Return holen sollte. Ein unabhaengiger Pruefagent hat ihn
	// als BEWEISBAR WIRKUNGSLOS entlarvt, und die Rechnung stimmt: angewandt wird in Pass 2
	//   q_i = 6 w_i (c_i . u_s) + w_i * alpha  mit  alpha = -6 (S1 . u_s)/S0 ,
	// also q_i = 6 w_i (c_i - c_q) . u_s. Fuer EINEN Link ist c_q = c_1, damit q_i IDENTISCH NULL --
	// fuer jedes u_s. G' = 0 war also keine numerische Entartung, sondern die wahre Aussage ueber den
	// unter der Massen-Nebenbedingung erreichbaren Unterraum. Schlimmer: mit zurueckgesetzten
	// Rohmomenten haette phi1 = P1 + G11roh*s1 eine Wandkraft in fac_tau[1..3] gebucht, die
	// nachweislich NICHT angewandt wurde -- K2 waere besser geworden, ohne dass sich am Feld etwas
	// aendert. Genau die Selbsttaeuschung, gegen die die Abnahme gebaut ist. Der Freiheitsgrad muss
	// aus der GEOMETRIE kommen (ELIBB, q-gewichteter Wandabstand), nicht aus dem Zuruecknehmen einer
	// Identitaet. Bestaetigt auch empirisch: ab_45_kontrolle und ab_45_a2fall sind in jeder
	// gedruckten Zahl gleich.
)+"#ifdef FAC_REK_R3"+R(
	// ★ 23.09. abends: G11roh hier einfangen statt spaeter darauf zuzugreifen. Der Gate-Block liegt
	// mehrere hundert Zeilen und mehrere Praeprozessorzweige weiter; eine Scope-Annahme ueber diese
	// Distanz ist in einem R()-String nicht beweisbar, und OpenCL C ist C99 -- der Lauf staerbe mit
	// error -11. Dasselbe Muster wie bei rek_dux/rek_rho.
)+"#endif"+R( // FAC_REK_R3
	const float G11roh=G11, G22roh=G22, Snnroh=Snn; // Rohmomente VOR dem ALPHA2-Downdate -- Snnroh (09.09.2026) fuer den Rauschboden des Vollrangtests, s. dort -- fuer den Slot-13-Split (Einzellink-diagonal gegen c-parallel-n; Planungsagent 1a)
)+"#ifdef FAC_REK_R3"+R(
	rek_g11 = G11roh;
)+"#endif"+R( // FAC_REK_R3
)+"#ifdef FACETTEN_ALPHA2"+R(
	// ★ J4-alpha Stufe 2 (Plan 2026-08-17): symmetrisches Rang-1-Downdate G' = 6 Sum w (c-cq)(c-cq)^T
	// mit cq = S1/S0 -- eine Kovarianz, garantiert PSD. Der Solve erreicht sein Impulsziel damit
	// INKLUSIVE des alpha-Terms exakt (Sum q c = G'*u_s), und die sn-Nullung nullt die TATSAECHLICHE
	// Normalinjektion inkl. alpha. Einzellink-Facette: G' == 0 analytisch -> faellt sauber in
	// Slot-13-Return (Entkopplung, nicht Slot 15!) -- statische Einzellink-Population; Waechter haelt eps*G-Rauschen aus der Kaskade
	// (J4-Flicker-Lehre: absolute Schwellen lagen exakt auf dem Rauschen).
	if(S0>0.0f) {
		const float G11r=G11, G22r=G22, Snnr=Snn;
		const float B1=S1x*t1x+S1y*t1y+S1z*t1z, B2=S1x*t2x+S1y*t2y+S1z*t2z, Bn=S1x*nx+S1y*ny+S1z*nz;
		const float Dd=6.0f/S0;
		G11 -= Dd*B1*B1; G22 -= Dd*B2*B2; G12 -= Dd*B1*B2;
		Sn1 -= Dd*B1*Bn; Sn2 -= Dd*B2*Bn; Snn -= Dd*Bn*Bn;
		if(G11<1e-4f*G11r) { G11=0.0f; G12=0.0f; Sn1=0.0f; } // Ausloeschungswaechter: t1-Richtung degeneriert -> Kaskade
		if(G22<1e-4f*G22r) { G22=0.0f; G12=0.0f; Sn2=0.0f; }
		if(Snn<1e-4f*Snnr) Snn=0.0f; // Entkopplungs-Gate (Snn<1e-8) uebernimmt
	}
)+"#endif"+R( // FACETTEN_ALPHA2
)+"#ifdef FACETTEN_PEMA"+R(
	// ★ Beidseitige Filterung (FACETTEN-IMEM-ANALYSE.md, Weg A): EMA auf P-Vektor UND Abtast-u
	// (xyz-Rahmen); geloest wird gegen die GEFILTERTEN Groessen -> Phi(t) = T + P_prime(t): das Ziel
	// exakt im Mittel, die Fluktuation laeuft als natuerlicher BB-Austausch durch (Schumann-Klasse,
	// Asmuth Gl. 29/30 -- er filtert den EINGANG, nie die Loesung). s wird glatt: Klemm-Rektifizierer
	// und nu_t-Pumpe (die J3-Rueckkopplungsschleife) sind konstruktiv aus.
	{
		const uxx e = 6ul*(uxx)fid;
		if(fac_tau_cnt[fid]==0u) { // Warmstart: erster Besuch uebernimmt Momentanwerte statt 0
			// LATENT (Audit 1/3, revidiert 27.08. Buchungsschluss): cnt zaehlt seit dem Schluss JEDEN
			// gebuchten Besuch (auch Rueckfall) -- der Warmstart re-seedet damit nur noch beim ersten
			// Besuch; unter PEMA x SATGATE baut der Filter jetzt ab dem ersten Rueckfall Historie auf
			// (Feld im PEMA-Arm NICHT bitgleich zum Vor-Schluss-Stand; PEMA ist widerlegter Legacy-Arm).
			fac_pu[e]=Pvx; fac_pu[e+1ul]=Pvy; fac_pu[e+2ul]=Pvz;
			fac_pu[e+3ul]=uxn; fac_pu[e+4ul]=uyn; fac_pu[e+5ul]=uzn;
		} else {
			const float ap = def_fac_pema;
			fac_pu[e]    = fma(ap, Pvx-fac_pu[e],     fac_pu[e]);
			fac_pu[e+1ul]= fma(ap, Pvy-fac_pu[e+1ul], fac_pu[e+1ul]);
			fac_pu[e+2ul]= fma(ap, Pvz-fac_pu[e+2ul], fac_pu[e+2ul]);
			fac_pu[e+3ul]= fma(ap, uxn-fac_pu[e+3ul], fac_pu[e+3ul]);
			fac_pu[e+4ul]= fma(ap, uyn-fac_pu[e+4ul], fac_pu[e+4ul]);
			fac_pu[e+5ul]= fma(ap, uzn-fac_pu[e+5ul], fac_pu[e+5ul]);
		}
		// Basis/Ziel/P aus den GEFILTERTEN Groessen neu (ueberschreibt die Momentanwerte oben):
		const float ubx=fac_pu[e+3ul], uby=fac_pu[e+4ul], ubz=fac_pu[e+5ul];
		const float undb = nx*ubx+ny*uby+nz*ubz;
		const float utxb=ubx-undb*nx, utyb=uby-undb*ny, utzb=ubz-undb*nz;
		const float utb = sqrt(utxb*utxb+utyb*utyb+utzb*utzb);
		if(utb<1e-6f) { if(t%def_zaehl_takt==0ul) atomic_inc(&hits[17]); return (float3)(0.0f,0.0f,0.0f); } // IR3-Audit M2: KEIN stiller Rueckfall in den widerlegten Instantan-Modus -- BB belassen, Slot 17 zaehlt (Staupunkt-/Abloesezellen)
		{
			t1x=utxb/utb; t1y=utyb/utb; t1z=utzb/utb;
			t2x=ny*t1z-nz*t1y; t2y=nz*t1x-nx*t1z; t2z=nx*t1y-ny*t1x;
			const float utb_wm = utb*def_fac_utkorr; const float Yb = utb_wm*((2.0f*yw)*def_fac_Y); // ★ Kernel-Audit MITTEL: UTKORR auch auf der PEMA-Kette (war wirkungslos unter PEMA)
			const float upb = wf_spalding_uplus(Yb);
			const float utaub = utb_wm/upb;
			float twb = rhon*utaub*utaub;
			const float twmb = 0.5f*rhon*utb;
			if((twb>twmb||twb*faca>twmb)&&t%def_zaehl_takt==0ul) atomic_inc(&hits[8]); // Slot 8: Klemmen der ANGEWANDTEN Kette (Audit 1/3 -- vorher zaehlte die verworfene Kopf-Kette)
			if(twb>twmb) twb = twmb;
			twe = fmin(twb*faca, twmb);
			ut = utb; // Klemmskalen folgen der gefilterten Basis
			// Momente in der NEUEN Basis (G/Sn haengen an t-hat) -- zweiter Pass ueber L:
			G11=0.0f; G22=0.0f; G12=0.0f; Sn1=0.0f; Sn2=0.0f; P1=0.0f; P2=0.0f;
			for(uint i2=1u; i2<def_velocity_set; i2++) {
				const uint ib2 = (i2%2u==1u) ? i2+1u : i2-1u;
				if((flags[j[ib2]]&TYPE_BO)!=TYPE_S) continue;
				const float cx2=c(i2), cy2=c(def_velocity_set+i2), cz2=c(2u*def_velocity_set+i2);
				const float wi2=w(i2);
				const float ct1b=cx2*t1x+cy2*t1y+cz2*t1z, ct2b=cx2*t2x+cy2*t2y+cz2*t2z, cnb=cx2*nx+cy2*ny+cz2*nz;
				G11 = fma(6.0f*wi2, ct1b*ct1b, G11); G22 = fma(6.0f*wi2, ct2b*ct2b, G22); G12 = fma(6.0f*wi2, ct1b*ct2b, G12);
				Sn1 = fma(6.0f*wi2, ct1b*cnb, Sn1); Sn2 = fma(6.0f*wi2, ct2b*cnb, Sn2);
			}
			P1 = fac_pu[e]*t1x+fac_pu[e+1ul]*t1y+fac_pu[e+2ul]*t1z; // GEFILTERTES P in der neuen Basis
			P2 = fac_pu[e]*t2x+fac_pu[e+1ul]*t2y+fac_pu[e+2ul]*t2z;
)+"#ifdef FACETTEN_ALPHA2"+R(
		if(S0>0.0f) { // Downdate in der GEFILTERTEN Basis wiederholen; Snn ist basisunabhaengig und oben bereits downgedatet
			const float G11q=G11, G22q=G22;
			const float B1=S1x*t1x+S1y*t1y+S1z*t1z, B2=S1x*t2x+S1y*t2y+S1z*t2z, Bn=S1x*nx+S1y*ny+S1z*nz;
			const float Dd=6.0f/S0;
			G11 -= Dd*B1*B1; G22 -= Dd*B2*B2; G12 -= Dd*B1*B2;
			Sn1 -= Dd*B1*Bn; Sn2 -= Dd*B2*Bn;
			if(G11<1e-4f*G11q) { G11=0.0f; G12=0.0f; Sn1=0.0f; }
			if(G22<1e-4f*G22q) { G22=0.0f; G12=0.0f; Sn2=0.0f; }
		}
)+"#endif"+R( // FACETTEN_ALPHA2
		}
	}
)+"#endif"+R( // FACETTEN_PEMA
)+"#ifndef FACETTEN_UW"+R(
	// ★ N4 GEPRUEFT UND VERWORFEN (24.09.): ein #error-Waechter "FAC_REK_S2 ohne FAC_REK_R3" ist
	// hier NICHT baubar. Am Zeilenanfang trifft ihn der C++-Praeprozessor (sofortiger Bauabbruch),
	// und als String-Literal zerreisst ihn get_opencl_c_code(): das ersetzt JEDES Leerzeichen
	// durch einen Zeilenumbruch und repariert nur #ifdef/#ifndef/#define/#undef/#if/#elif/#pragma,
	// nicht #error. Der Fall ist ohnehin strukturell ausgeschlossen: FAC_REK_S2 wird nur bei
	// s_fac_rek>=3 emittiert, FAC_REK_R3 schon bei >=2 (lbm.cpp, JIT-Zeile).
)+"#ifdef FAC_REK_S2"+R(
	// ★★ S2: hier entsteht die Amplitude des NAECHSTEN Schritts. rho ist das geklemmte rhon,
	// dasselbe, das die Gewichte multipliziert -- und es steht jetzt im NENNER, die Klemme wirkt
	// also verstaerkend statt daempfend (an 85 % der Marken liegt rhon auf der Klemme). Slot 394.
	// Die Schranke ist KEIN neuer Knopf: tw_max = 0,5*rhon*ut ist derselbe Ausdruck, gegen den
	// twe oben schon geklemmt wird. Reisst sie, faellt die Zelle auf reines Bounce-Back zurueck
	// (du := 0) statt zu klemmen -- SATGATE-Logik, Befund G8: die geklemmte Anwendung hat einen
	// vorzeichen-definiten Bias, der Rueckfall hat keinen.
	if(rek_gate) {
		// ★★ H1 (24.09., eigener Verdacht, vom Pruefagenten bestaetigt und beziffert): P1 traegt an
		// dieser Stelle das Df des LAUFENDEN Schritts mit -- die Injektion lief davor. Exakt gilt
		// P1 = P1_vor + rek_dp1, und rek_dp1 ist die Nachbildung ueber dieselbe Linkmenge. Ohne die
		// Ruecknahme lautet die Iteration e_{n+1} = e* - G11roh*e_n, und der Fixpunkt liegt bei
		// e*/(1+G11roh) -- der Arm verfehlt sein eigenes Ziel AUCH NACH KONVERGENZ, um 25 % bei
		// G11roh = 1/3 und 40-57 % an Treppenzellen. Mit der Ruecknahme entfaellt die algebraische
		// Rueckkopplung vollstaendig (e_{n+1} = e*, deadbeat); der physikalische Ein-Schritt-Lag
		// bleibt und wird von Slot 393 gemessen.
		// ★★ H5: die ELIBB-Tangentialbuchung +2*Dp_t (weiter unten, hinter fw) gehoert in die
		// Zielgleichung. Tatsaechlich ist fw.t1 = -P1_vor - rho*d + 2*(elibb_dp.t1); ohne den
		// dritten Summanden zielt S2 auf die falsche Groesse. Am Fahrzeug ist ELIBB nicht
		// abschaltbar (ohne ihn verzehnfacht sich die Reibung), der Term ist dort kein Rest.
		// ★★★ S1 (24.09.2026 nachmittags): DAS REINE MODELLZIEL. Hier stand bis eben die
		// BUCHUNGSIDENTITAET -def_fac_tau*twe - (P1 - rek_dp1) + 2*(elibb_dp.t1). Die ist gemessen
		// gekippt (Ub +143 %, cf 0) -- aber NICHT, weil eine Zellquelle auf R1 nicht tragen kann:
		// CFD_FAC_KRAFT speist DASSELBE R1 lagfrei an einer Obermenge ein und haelt den Antrieb
		// auf 1,6 %. Was kippt, ist die Kombination aus (a) P1 im Ziel und (b) Lag 1.
		// P1 ist an Treppen der DEVIATORISCHE Austausch und im FLAECHENINTEGRAL exakt 0 -- ueber
		// die ganze Wand hebt -P1 sich weg, ueber die markierte 1/3-Teilmenge NICHT. Eine
		// Zellquelle darauf ist eine stehende Koerperkraft; der Akkumulator hat +64,63 je Schritt
		// gegen ein Budget von 0,998 gemessen, also fast genau |P1|/twe = 67,85 dieser Menge.
		// DESHALB JETZT: nur der Modellanteil. Kein P1, kein ELIBB-Glied.
		// STRUKTURELLE ABNAHME -- ★ BERICHTIGT 24.09. nachmittags (Pruefagent H1/H2). Hier stand:
		//   "twe = fmin(tw*faca, 0,5*rhon*ut) => |s2_d1| <= def_fac_tau*0,5*ut, also kann die Schranke
		//    fuer def_fac_tau <= 1 NIE greifen. SLOT 395 MUSS EXAKT 0 SEIN."
		// Der Schluss stimmt ALGEBRAISCH, die Umsetzung stand in float daneben: s2_max = 0.5f*ut trug
		// null Rundungen, s2_d1 = fl(0.5f*rhon*ut)/rhon deren zwei. Genau dort, wo die twe-Klemme bindet
		// (Slot 8), ueberschritt s2_d1 die Schranke um 1-2 ulp und die Zelle fiel STILL auf Bounce-Back
		// zurueck. Der frueher gemessene Slot 400 = 9 ist genau diese Kante.
		// JETZT wird im twe-Bereich verglichen, mit BUCHSTAEBLICH demselben Ausdruck wie das fmin in
		// 2328 und 2406 (0.5f*rhon*ut, gleiche Assoziation) -- damit ist die Kante exakt.
		// FOLGE, die man wissen muss: die Schranke ist unter S1 damit STRUKTURELL redundant. Slot 395
		// ist eine STOLPERDRAHT-Null, KEINE bestandene Messung. Er feuert nur, wenn jemand die
		// tw_max-Klemme entfernt (wie beim MOZ-Tausch 2371 schon einmal geschehen).
		// DESHALB: SLOT 395 IST NUR ZUSAMMEN MIT SLOT 8 DEUTBAR -- Slot 8 sagt, ob der Pfad ueberhaupt lief.
		// ★ DIAGNOSELEITER (CFD_FAC_REK_LEITER, 24.09. spaet). Der Faktor steht HIER und nicht weiter
		// unten, damit die Schranke s2_ok die SKALIERTE Amplitude sieht: sonst koennte die Leiter
		// unbemerkt ueber 0,5*u_t hinausschieben. Bei 1.0f ist die Multiplikation verlustfrei
		// (IEEE754), der Arm also bitgleich -- das wird als Sprosse 1 der Leiter GEMESSEN, nicht
		// behauptet. Wirkpfadbeleg ist das Histogramm [403..407]: es MUSS mit der Leiter wandern.
		const float s2_r1 = -def_fac_tau*twe*def_fac_rek_leiter;
		const float s2_d1 = s2_r1/rhon;
		const float s2_alt = fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+4ul];
		const bool s2_ok = (fabs(s2_r1)<=def_fac_tau*(0.5f*rhon*ut)); // ★ H2: derselbe Ausdruck wie das fmin, keine Division, keine ulp-Kante
		const float s2_neu = s2_ok ? s2_d1 : 0.0f;
		fac_nb[def_nb_stride*(ulong)fid+def_nb_roff+4ul] = s2_neu;
		if(t%def_zaehl_takt==0ul) {
			// [393] SCHRITTFLUKTUATION der Amplitude. ★ BERICHTIGT 24.09. (Pruefagent H3): hier stand
			// "KONVERGENZ ... faellt sie nicht, ist der Lag-1-Kreis nicht kontrahiert". Das galt fuer das
			// ALTE Ziel mit P1 (e_{n+1} = e* - G11roh*e_n). Seit S1 haengt s2_neu nur noch an twe und rhon
			// DESSELBEN Schritts -- es gibt keine Rueckkopplung mehr, also auch nichts zu kontrahieren.
			// Was der Zaehler jetzt misst: wie stark die Amplitude von Schritt zu Schritt schwankt, und
			// damit DIREKT, wie weit die ANGEWANDTE (= die des Vorschritts, Lag 1) von der richtigen
			// abweicht. 63,8 % gemessen heisst: an fast zwei Dritteln der Marken ist der Lag-Fehler > 5 %.
			// Das bleibt ein Grund, den Lag zu entfernen -- aber es ist KEIN Konvergenzbefund.
			const float s2_dd = fabs(s2_neu-s2_alt);
			if(s2_dd>0.05f*fmax(fabs(s2_neu), 1.0E-30f)&&hits[393]<0xF0000000u) atomic_inc(&hits[393]);
			// [394] die rho-Klemme VERSTAERKT hier, statt zu daempfen -- eigene Klasse, eigener Zaehler.
			if(rhon<=0.5f&&hits[394]<0xF0000000u) atomic_inc(&hits[394]);
			// [395] die Schranke hat gegriffen, die Zelle faellt auf Bounce-Back zurueck.
			if(!s2_ok&&hits[395]<0xF0000000u) atomic_inc(&hits[395]);
			// [396] RICHTUNG der Schrittaenderung. ★ ERSETZT 24.09. (Pruefagent H1): hier stand das
			// Vorzeichen von R1. Unter S1 ist s2_r1 = -def_fac_tau*twe mit twe >= 0 lueckenlos und
			// def_fac_tau in {0;1} (drei Zuweisungen, alle Literale: setup.cpp:11537/11546/11661 -- ein
			// CFD_FAC_TAU gibt es NICHT). Also s2_r1 <= 0 IMMER: der Zaehler KONNTE nicht feuern, und der
			// Gegenphasentest verglich 0 % gegen 0 % und meldete zwangslaeufig "keine Periode-2-Mode".
			// Das ist die Klasse, die dieses Projekt an Slot 124/125 und an Slot 331 schon bezahlt hat.
			// JETZT: bei einer Periode-2-Mode steigt die Amplitude in der einen Paritaet und faellt in der
			// anderen, also 396/397 -> 100 % und 401/402 -> 0 % (oder umgekehrt). Bei reiner Fluktuation
			// liegen BEIDE bei rund 50 %. Das ist der Detektor, den HOCH-2 gemeint hat.
			if(s2_neu>s2_alt&&hits[396]<0xF0000000u) atomic_inc(&hits[396]);
			// [397] WIRKPFAD und Nenner fuer alle vier.
			if(hits[397]<0xF0000000u) atomic_inc(&hits[397]);
			// [403..407] AMPLITUDENGROESSE -- ★ NEU 24.09. (Pruefagent H4), das entscheidende fehlende
			// Instrument. Unter S2 ist CFD_FAC_REK_EPS zwingend 0 (setup.cpp:786), deshalb schweigen BEIDE
			// Groessenwaechter: setup.cpp:857 (harte Sperre 1e-6, "der Hub ueberlebt store_f nicht") und
			// setup.cpp:863 (Messhub 1e-4, "darunter koennen alle Zaehler gruen sein, ohne dass der Hub den
			// naechsten Zeitschritt erreicht -- sie messen registerseitig"). Beide haengen an rek_eps_b>0.
			// Fuer die LAUFZEITamplitude gab es damit keinen einzigen Waechter -- und genau sie liegt nach
			// der Herleitung bei |du| = twe/rhon ~ 4,5e-6..1,35e-5 (twe = 6,733998e-06 gemessen am kipp26,
			// setup.cpp:766; rhon in [0,5;1,5]), also 7- bis 22-fach UNTER dem Messhub.
			// Die Faechergrenzen SIND diese projekteigenen Schwellen: der Lauf sagt damit selbst, ob er im
			// stillen Band arbeitet. Nenner ist 397 (gleiches Gate).
			// ACHTUNG BERECHNETER INDEX: ein Literal-Grep nach "hits[404]" findet diese Slots NICHT --
			// dieselbe Falle wie die SGS_DIAG-Faecher 30..48, die zweimal bezahlt wurde.
			const float s2_b = fabs(s2_neu);
			const uint s2_f = (s2_b<1.0E-6f) ? 403u : ((s2_b<1.0E-5f) ? 404u : ((s2_b<1.0E-4f) ? 405u : ((s2_b<1.0E-3f) ? 406u : 407u)));
			if(hits[s2_f]<0xF0000000u) atomic_inc(&hits[s2_f]);
		}
		// ★★ GEGENPHASE (HOCH-2, 24.09. vormittags gebaut -- ★ NEU GEFASST 24.09. nachmittags).
		// Bauanlass: die Zaehler oben tasten alle dieselbe Paritaet ab (def_zaehl_takt gerade, Zaehl-
		// schritte 0, takt, 2*takt ...), waehrend die dort GELESENE Amplitude vom Schritt davor stammt,
		// also von einer ungeraden. Eine Periode-2-Mode waere dadurch konstruktiv unsichtbar.
		// WAS DIE ERSTE FASSUNG NICHT LEISTETE: sie spiegelte 395 und 396, und BEIDE sind unter S1
		// konstruktiv still -- 395 strukturell redundant, 396 wegen s2_r1 <= 0 unmoeglich. Der Vergleich
		// lief also 0 % gegen 0 % und meldete zwangslaeufig "keine Periode-2-Mode": eine Tautologie, die
		// wie eine bestandene Abnahme aussah. Gemessen wurde sie als 99,7 % gegen 99,7 % (0,0 pp).
		// JETZT spiegelt 401 den RICHTUNGSzaehler 396 (steigt/faellt). Weichen 401/402 und 396/397 stark
		// voneinander ab, schwingt die Amplitude -- und dann ist keine Kraftzahl dieses Arms deutbar.
		// ★ M3 OFFEN: dass def_zaehl_takt GERADE ist, waechtert nirgends jemand. zaehl_takt() skaliert
		// mit dx (lbm.cpp:636); bei dx = 3,75 mm ergibt CFD_ZAEHL_TAKT=1000 den Wert 1067 -- ungerade,
		// dann treffen ==0 und ==1 beide Paritaeten und dieser ganze Block ist bedeutungslos.
		// Der Waechter gehoert an die Emission von def_zaehl_takt in lbm.cpp.
		if(t%def_zaehl_takt==1ul) {
			if(!s2_ok&&hits[400]<0xF0000000u) atomic_inc(&hits[400]);
			if(s2_neu>s2_alt&&hits[401]<0xF0000000u) atomic_inc(&hits[401]); // ★ H1: spiegelt jetzt 396 (Richtung), war das tote R1-Vorzeichen
			if(hits[402]<0xF0000000u) atomic_inc(&hits[402]);
		}
	}
)+"#endif"+R( // FAC_REK_S2
	const float R1 = -def_fac_tau*twe - P1, R2 = -P2; // Ziel: (-def_fac_tau*twe, 0, 0_normal) -- 3x3-Plan Gl. 18
)+"#endif"+R( // FACETTEN_UW -- eigener Guard an der URSPRUENGLICHEN Stelle, damit der Kontrollarm bei CFD_FAC_UW=0 QUELLTEXTIDENTISCH bleibt (Pruefbefund M2)
	float s1=0.0f, s2=0.0f, sn=0.0f;
	bool rueckfall=false; // ★ BUCHUNGSSCHLUSS (Baustein 2/1, 27.08.): Rueckfaelle steigen nicht mehr per return aus, sondern buchen mit s=0 (P-only)
	float res2=0.0f;
	uint zweig=0u; // ★ KREUZTABELLE 04.09. abends: Solve-Zweig der REALEN Kaskade, 1=[78] 2=[79] 3=[12] 4=[14]/[80]; nur Zaehler, kein Float
)+"#ifdef FAC_R1Q"+R(
	bool r1q_pinv=false;
	float r1q_a=0.0f;
	float r1q_b=0.0f;
)+"#endif"+R( // FAC_R1Q
)+"#ifndef FACETTEN_UW"+R(
)+"#ifdef FACETTEN_MASSE_X"+R(
	// ★ ARM X (CFD_FAC_MASSE_ALLE=3, 04.09.2026): Rueckfall-Entscheid im SCHATTEN wie unter ALPHA2 --
	// Downdate auf KOPIEN, Entkopplungs-Gate, Kaskade, Schatten-Solve, Gates auf dem Schatten-s. Die
	// reale Kaskade darunter loest weiter gegen die ROHEN Momente. Jede Bedingung ist Zeichen fuer
	// Zeichen die der Basis (2162-2179, 2257-2325, 2335, 2342, 2350-2360). KEIN hits[] hier -- sonst
	// zaehlt die Kaskade doppelt und die Abnahme 78+79+12+13+14+15 == Wirkpfad bricht.
	bool a2_rueckfall=false;
	{
		float G11s=G11, G22s=G22, G12s=G12, Sn1s=Sn1, Sn2s=Sn2, Snns=Snn;
		if(S0>0.0f) {
			const float G11rs=G11s, G22rs=G22s, Snnrs=Snns;
			const float B1=S1x*t1x+S1y*t1y+S1z*t1z, B2=S1x*t2x+S1y*t2y+S1z*t2z, Bn=S1x*nx+S1y*ny+S1z*nz;
			const float Dd=6.0f/S0;
			G11s -= Dd*B1*B1; G22s -= Dd*B2*B2; G12s -= Dd*B1*B2;
			Sn1s -= Dd*B1*Bn; Sn2s -= Dd*B2*Bn; Snns -= Dd*Bn*Bn;
			if(G11s<1e-4f*G11rs) { G11s=0.0f; G12s=0.0f; Sn1s=0.0f; }
			if(G22s<1e-4f*G22rs) { G22s=0.0f; G12s=0.0f; Sn2s=0.0f; }
			if(Snns<1e-4f*Snnrs) Snns=0.0f;
		}
		float s1s=0.0f, s2s=0.0f, res2s=0.0f;
		const float kops = Sn1s*Sn1s+Sn2s*Sn2s;
		if(Snns<1e-8f||kops<=1e-6f*Snns*(G11s+G22s)) {
		const float dets = G11s*G22s - G12s*G12s;
		if(dets>=1e-4f*G11s*G22s&&G11s>=1e-8f&&G22s>=1e-8f) { s1s=(R1*G22s-R2*G12s)/dets; s2s=(R2*G11s-R1*G12s)/dets; }
)+"#ifdef FACETTEN_LSQ"+R(
		else if(G11s>=1e-8f) { const float d=fma(G11s,G11s,G12s*G12s); s1s=(d>0.0f)?(G11s*R1+G12s*R2)/d:0.0f; s2s=0.0f; res2s=fabs(G12s*s1s-R2); }
		else if(G22s>=1e-8f) { const float d=fma(G22s,G22s,G12s*G12s); s2s=(d>0.0f)?(G12s*R1+G22s*R2)/d:0.0f; s1s=0.0f; res2s=fabs(G22s*s2s-R2); }
)+"#else"+R(
		else if(G11s>=1e-8f) { s1s=R1/G11s; s2s=0.0f; res2s=fabs(G12s*s1s-R2); }
		else if(G22s>=1e-8f) { s1s=0.0f; s2s=R2/G22s; res2s=fabs(G22s*s2s-R2); }
)+"#endif"+R( // FACETTEN_LSQ
		else a2_rueckfall=true;
		} else {
		const float Gt11s = G11s - Sn1s*Sn1s/Snns, Gt22s = G22s - Sn2s*Sn2s/Snns, Gt12s = G12s - Sn1s*Sn2s/Snns;
		const float detts = Gt11s*Gt22s - Gt12s*Gt12s;
		if(detts>=1e-4f*Gt11s*Gt22s&&Gt11s>=1e-4f*G11s&&Gt22s>=1e-4f*G22s&&Gt11s>=1e-8f&&Gt22s>=1e-8f) { s1s=(R1*Gt22s-R2*Gt12s)/detts; s2s=(R2*Gt11s-R1*Gt12s)/detts; }
)+"#ifdef FACETTEN_PINV"+R(
		else if(Gt11s>=0.0f&&Gt22s>=0.0f&&Gt11s+Gt22s>=1e-4f*(G11s+G22s)&&Gt11s+Gt22s>=1e-8f) { const float tr=Gt11s+Gt22s, it2=1.0f/(tr*tr); s1s=(Gt11s*R1+Gt12s*R2)*it2; s2s=(Gt12s*R1+Gt22s*R2)*it2; res2s=fabs(fma(Gt12s,s1s,Gt22s*s2s)-R2); }
)+"#elif defined(FACETTEN_LSQ)"+R(
		else if(Gt11s>=1e-4f*G11s&&Gt11s>=1e-8f) { const float d=fma(Gt11s,Gt11s,Gt12s*Gt12s); s1s=(d>0.0f)?(Gt11s*R1+Gt12s*R2)/d:0.0f; s2s=0.0f; res2s=fabs(Gt12s*s1s-R2); }
		else if(Gt22s>=1e-4f*G22s&&Gt22s>=1e-8f) { const float d=fma(Gt22s,Gt22s,Gt12s*Gt12s); s2s=(d>0.0f)?(Gt12s*R1+Gt22s*R2)/d:0.0f; s1s=0.0f; res2s=fabs(fma(Gt12s,s1s,Gt22s*s2s)-R2); }
)+"#else"+R(
		else if(Gt11s>=1e-4f*G11s&&Gt11s>=1e-8f) { s1s=R1/Gt11s; s2s=0.0f; res2s=fabs(Gt12s*s1s-R2); }
		else if(Gt22s>=1e-4f*G22s&&Gt22s>=1e-8f) { s1s=0.0f; s2s=R2/Gt22s; res2s=fabs(fma(Gt12s,s1s,Gt22s*s2s)-R2); }
)+"#endif"+R( // FACETTEN_PINV / FACETTEN_LSQ
		else a2_rueckfall=true;
		}
)+"#ifdef FACETTEN_QUERGATE"+R(
		if(!a2_rueckfall&&def_fac_tau>0.0f&&res2s>def_fac_tau*twe) a2_rueckfall=true;
)+"#endif"+R( // FACETTEN_QUERGATE
)+"#ifdef FACETTEN_SATGATE"+R(
		if(!a2_rueckfall&&(fabs(s1s)>2.0f*def_fac_budget*ut||fabs(s2s)>def_fac_budget*ut)) a2_rueckfall=true;
		if(!a2_rueckfall&&Snns>=1e-8f&&(Sn1s*Sn1s+Sn2s*Sn2s)>1e-6f*Snns*(G11s+G22s)) {
			const float sns = -(Sn1s*s1s+Sn2s*s2s)/Snns;
			if(fabs(sns)>def_fac_budget_sn*ut) a2_rueckfall=true;
		}
)+"#endif"+R( // FACETTEN_SATGATE
	}
)+"#endif"+R( // FACETTEN_MASSE_X // ★ 2026-08-25 Restfehler in QUERrichtung t2 (Ziel dort ist 0); nur die Rueckfaelle fuellen ihn
	// ★ 3x3-Iteration (FACETTEN-IMEM-3X3.md): Entkopplungs-Gate (Gl. 20). Entkoppelt laeuft
	// WOERTLICH der bisherige 2x2-Pfad (Bitgleichheit der ebenen Waende); gekoppelt wird die
	// Normalinjektion per Schur-Elimination exakt genullt (I2-Durchfallursache bei 26,6 Grad).
	const float kop = Sn1*Sn1+Sn2*Sn2;
	if(Snn<1e-8f||kop<=1e-6f*Snn*(G11+G22)) {
	const float det = G11*G22 - G12*G12;  // Degenerationskaskade (Gl. 8, Nachpruefer-Befund 2/3 geschlossen)
	if(det>=1e-4f*G11*G22&&G11>=1e-8f&&G22>=1e-8f) { s1=(R1*G22-R2*G12)/det; s2=(R2*G11-R1*G12)/det;
		// ★ 03./04.09.2026 Slot 78: der EXAKTE entkoppelte Solve. Dieser Zweig war der EINZIGE im ganzen
		// Solver ohne Zaehler -- und genau das hat den Etikettenfehler in B78 erzeugt: der Vollranganteil
		// musste per Subtraktion geschaetzt werden, dabei griff ich zum naechstbesten Slot (14) und der ist
		// der gekoppelte SKALAR-Rueckfall. Ein fehlender Zaehler hat hier nicht nur eine Messung verhindert,
		// sondern eine FALSCHE erzeugt. Jetzt ist die Kaskade lueckenlos: 78+79+12+13+14+15 == Wirkpfad-Rest.
		if(t%def_zaehl_takt==0ul&&hits[78]<0xF0000000u) atomic_inc(&hits[78]); zweig=1u; }
	// ★★ 2026-08-25 KLEINSTE-QUADRATE-RUECKFALL (CFD_FAC_LSQ, Default AUS -- Pruefbefund 4-A: es ist eine MODELLAENDERUNG, keine Fehlerkorrektur). Der alte Skalar-Rueckfall
	// s1 = R1/G11 erzwingt das Ziel in Richtung 1 EXAKT und ignoriert die zweite Gleichung ganz.
	// Ist G11 fast entartet, wird s1 riesig -- und weil G12 dabei NICHT klein sein muss, schleppt
	// die Loesung dann Phi2 = G12*s1 mit, gemessen 10^2 bis 10^3 mal twe (26-Grad-Wand, Slot 14).
	// Richtig ist die kleinste-Quadrate-Loesung auf dem erreichbaren Unterraum {s1*(G11,G12)}:
	// s1 = (G11*R1 + G12*R2)/(G11^2 + G12^2). Bei G12 = 0 ist sie WORTGLEICH die alte; bei G11 -> 0
	// laeuft sie gegen R2/G12 statt gegen unendlich. Das Residuum steht dann senkrecht auf dem
	// Erreichbaren -- das ist die Definition von "so nah am Ziel wie diese Linkmenge es zulaesst".
)+"#ifdef FACETTEN_LSQ"+R(
	else if(G11>=1e-8f) { const float d=fma(G11,G11,G12*G12); s1=(d>0.0f)?(G11*R1+G12*R2)/d:0.0f; s2=0.0f; res2=fabs(G12*s1-R2); if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[12]); atomic_inc(&hits[65]); } zweig=3u; }
	else if(G22>=1e-8f) { const float d=fma(G22,G22,G12*G12); s2=(d>0.0f)?(G12*R1+G22*R2)/d:0.0f; s1=0.0f; res2=fabs(G22*s2-R2); if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[12]); atomic_inc(&hits[65]); } zweig=3u; }
)+"#else"+R(
	else if(G11>=1e-8f) { s1=R1/G11; s2=0.0f; res2=fabs(G12*s1-R2); if(t%def_zaehl_takt==0ul) atomic_inc(&hits[12]); zweig=3u; } // Slot 12: Skalar-Fallback t1
	else if(G22>=1e-8f) { s1=0.0f; s2=R2/G22; res2=fabs(G22*s2-R2); if(t%def_zaehl_takt==0ul) atomic_inc(&hits[12]); zweig=3u; } // Slot 12: Skalar-Fallback t2
)+"#endif"+R( // FACETTEN_LSQ
	else { if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[13]); if(G11roh>=1e-8f||G22roh>=1e-8f) atomic_inc(&hits[27]); } rueckfall=true; } // Slot 13: kein tangential wirksamer Link; [27] = Teilmenge mit rohen Tangentialmomenten. ★ 03.09.2026 KORRIGIERT: die alte Bezeichnung "die ELIBB-heilbare Klasse" stammt vom 22.08. und ist seit der B2-REVISION vom 25.08. FALSCH. elibb_rekonstruiere iteriert ueber DIESELBE Linkmenge mit demselben Praedikat und aendert nur die WERTE fhn[i] -- G11/G22/G12/S0/S1/Snn sind mit und ohne ELIBB bitgleich, der Rang aendert sich NIE. Empirisch: w_nb lief MIT ELIBB (Slot 67 = 139.185.273) und Slot 27 steht trotzdem bei 15,23 % der Wandbesuche. Heilbar waere die Klasse nur mit der urspruenglich geplanten Blende u_W = u_s gewesen, und genau die wurde am 25.08. zurueckgenommen (sie ist bei q=0,5 die Identitaet, der Wandmodell-impuls waere an jeder ebenen Partie konstruktiv ausgefallen). Der wahre Grund ist Rang: das ALPHA2-Downdate macht aus dem zweiten Moment die KOVARIANZ der Linkrichtungen, bei einem Link ist sie null, und die Massenerhaltung erzwingt dort q_1 = 0 -- unter JEDEM Ansatz [06.09.: UEBERHOLT unter MASSE_ALLE -- dort ist q_1 != 0 erlaubt, es traegt allein J.n = 0; s. 3X3.md Gl. 28 Nachtrag]
	} else {
	// Schur-Reduktion (Gl. 19): Gt_ab = G_ab - Sn_a*Sn_b/Snn; RHS bleibt (R1,R2); sn = -(Sn*s)/Snn.
	const float Gt11 = G11 - Sn1*Sn1/Snn, Gt22 = G22 - Sn2*Sn2/Snn, Gt12 = G12 - Sn1*Sn2/Snn;
	const float dett = Gt11*Gt22 - Gt12*Gt12;
	// ★ J4-Befund #1 (Kugel): ABSOLUTE 1e-8-Schwellen lagen exakt auf dem float-Schur-Residuum
	// eps*G11 der Einzellink-Zellen -- 31 % der Kugelfacetten FLACKERTEN ins Rang-2 mit
	// G~=Rauschen, s=R/1e-8~1e6, Doppelklemme, permanente Injektion (belegt: 546 statt 485
	// Beitraeger). Jetzt RELATIV zu den Vor-Schur-Diagonalen: 3 Dekaden Marge zu eps beidseitig.
	// ★★ 09.09.2026 RAUSCHBODEN (CFD_FAC_DETEPS, Default 0 = bitidentisch). Fortsetzung der
	// J4-Lehre oben, eine Ebene tiefer. Fuer die EBENE Voxel-Linkmenge ist G' nach dem ALPHA2-
	// Downdate exakt (1/3)(I - m m^T) mit m = Voxelachse; das Schur-Komplement hat dann exakt
	// Rang 1 und dett ist ANALYTISCH NULL. Numerisch bleibt Rauschen: Snn entsteht bei 2190 als
	// Differenz zweier O(1)-Groessen und traegt den Absolutfehler eps*Snnroh; ueber Gt = G - Sn Sn/Snn
	// schlaegt er als eps*(G11+G22)*Snnroh/Snn in dett durch. Die relative Schranke 1e-4*Gt11*Gt22
	// faellt mit Gt11 = (1/3)sin^2(psi) gegen NULL und liegt bei kleinem Kippwinkel UNTER dem
	// Rauschen -- die Zelle nimmt dann einen Vollrangzweig, den es nicht gibt, und dividiert durch
	// Rauschen: s1 wird 1e4..1e7 mal u_t und beide Gates reissen zu Recht.
	// GEMESSEN 09.09. (8 mm, ph8_persist): 93,3 % aller Gate-Rueckfaelle kommen aus diesem Zweig
	// (Quote 45,8 %) gegen 2,8 % im PINV-Zweig darunter; Rueckfall gegen Kippwinkel springt bei
	// genau 1 Grad von 1,5 auf 92,7 % -- dort endet der Schutz des Waechters bei 2193.
	const float det_eps = def_fac_deteps*1.1920929e-7f*(G11+G22)*(Snnroh/fmax(Snn,1e-30f));
	if(dett>=1e-4f*Gt11*Gt22+det_eps&&Gt11>=1e-4f*G11&&Gt22>=1e-4f*G22&&Gt11>=1e-8f&&Gt22>=1e-8f) { s1=(R1*Gt22-R2*Gt12)/dett; s2=(R2*Gt11-R1*Gt12)/dett;
		if(t%def_zaehl_takt==0ul&&hits[79]<0xF0000000u) atomic_inc(&hits[79]); zweig=2u; } // ★ Slot 79: der EXAKTE gekoppelte Schur-Solve (Gegenstueck zu 78, s. dort)
)+"#ifdef FACETTEN_PINV"+R(
	// ★★ 04.09.2026 RANG-1-PSEUDOINVERSE (CFD_FAC_PINV, Default aus). Der gemessene Befund dahinter:
	// fuer die ebene Voxelwand ist nach dem ALPHA2-Downdate G' = (1/3)(I - n n^T), und die Schur-
	// Reduktion liefert dett == 0 und tr(Gt) == 1/3 EXAKT -- fuer jede Normale, jeden Azimut. Der
	// Vollrangzweig faellt korrekt durch. Die Skalarleiter darunter dividiert dann durch Gt11, also
	// durch den Anteil des WILLKUERLICH gewaehlten Basisvektors t1 (er zeigt entlang der Stroemung)
	// an der EINEN erreichbaren Richtung: Gt11 = (1/3)*t2z^2/(1-nz^2). Laeuft die Stroemung den Hang
	// hinauf, geht Gt11 gegen 0, waehrend der groesste Eigenwert konstant 1/3 bleibt. Die Schwelle
	// 1e-4 laesst damit eine Verstaerkung bis 1e4 durch, unmittelbar vor das harte Gate bei 2*ut.
	// BELEG (Klassentabelle za_basis, 8 mm): gleiche Linkzahl 5 -- y_w 0,50 gibt 11,7 % Rueckfall bei
	// s1/ut +0,654, y_w 0,51 gibt 88,9 % bei s1/ut +0,003. 0,01 Wandabstand kippt das Modell.
	// KUR: Moore-Penrose der symmetrisch-PSD Rang-1-Matrix. Gt = lambda*e*e^T mit lambda = tr, also
	// Gt^+ = Gt/tr^2 und s = (Gt*R)/tr^2 -- die Division laeuft ueber den GROESSTEN Eigenwert statt
	// ueber einen Diagonaleintrag. Fuer die dominante Klasse ist tr exakt 1/3 und kippungsunabhaengig;
	// die 1e4-Verstaerkung verschwindet konstruktiv statt statistisch.
	// NEBENWIRKUNG auf den zweiten Gate-Block: der Eigenvektor v ~ (t2z, -t1z) erfuellt Sn.v = 0 EXAKT
	// -- die pseudoinverse Loesung braucht KEINE Normalkompensation. sn wird strukturell klein, und
	// das sn-Gate (Slot 16) verliert seinen kuenstlichen Ausloeser.
	// Der Vollrangzweig oben bleibt Zeichen fuer Zeichen unangetastet -- die ebene Wand (kipp0 laeuft
	// entkoppelt) bleibt damit bitgleich.
	else if(Gt11>=0.0f&&Gt22>=0.0f&&Gt11+Gt22>=1e-4f*(G11+G22)&&Gt11+Gt22>=1e-8f) { // ★ D17: PSD-Haertung -- Rundungsnegative aus dem Schur-Komplement nicht "invertieren"
		const float tr=Gt11+Gt22, it2=1.0f/(tr*tr);
		s1=(Gt11*R1+Gt12*R2)*it2; s2=(Gt12*R1+Gt22*R2)*it2;
		res2=fabs(fma(Gt12,s1,Gt22*s2)-R2);
		if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[14]); if(hits[80]<0xF0000000u) atomic_inc(&hits[80]); } zweig=4u;
)+"#ifdef FAC_R1Q"+R(
		r1q_pinv = true;
		r1q_a = Gt11/tr;
		r1q_b = Gt12/tr;
)+"#endif"+R( // FAC_R1Q
	}
)+"#elif defined(FACETTEN_LSQ)"+R(
	else if(Gt11>=1e-4f*G11&&Gt11>=1e-8f) { const float d=fma(Gt11,Gt11,Gt12*Gt12); s1=(d>0.0f)?(Gt11*R1+Gt12*R2)/d:0.0f; s2=0.0f; res2=fabs(Gt12*s1-R2); if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[14]); atomic_inc(&hits[65]); } zweig=4u; } // Slot 14, kleinste Quadrate (s.o.)
	else if(Gt22>=1e-4f*G22&&Gt22>=1e-8f) { const float d=fma(Gt22,Gt22,Gt12*Gt12); s2=(d>0.0f)?(Gt12*R1+Gt22*R2)/d:0.0f; s1=0.0f; res2=fabs(fma(Gt12,s1,Gt22*s2)-R2); if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[14]); atomic_inc(&hits[65]); } zweig=4u; } // ★ 04.09. ZURUECKGENOMMEN (Kernel-Audit M1, selben Tag): hier stand kurzzeitig -R1. Das mischt die LINKE Seite von Gl. 2 mit der RECHTEN von Gl. 1 und faellt mit s1=0 auf |R2-R1| zusammen -- unabhaengig von Gt UND s. res2 ist per Definition das Gl.-2-Residuum (QUERGATE = Restfehler in Querrichtung t2), und im LSQ-Zweig ist es echt von null verschieden. Ein Mass fuer die Gl.-1-Luecke braucht eine EIGENE Groesse res1 mit eigenem Slot, nicht diese hier.
)+"#else"+R(
	else if(Gt11>=1e-4f*G11&&Gt11>=1e-8f) { s1=R1/Gt11; s2=0.0f; res2=fabs(Gt12*s1-R2); if(t%def_zaehl_takt==0ul) atomic_inc(&hits[14]); zweig=4u; } // Slot 14: gekoppelter Rang-2-Pfad
	else if(Gt22>=1e-4f*G22&&Gt22>=1e-8f) { s1=0.0f; s2=R2/Gt22; res2=fabs(fma(Gt12,s1,Gt22*s2)-R2); if(t%def_zaehl_takt==0ul) atomic_inc(&hits[14]); zweig=4u; } // ★ 04.09. ZURUECKGENOMMEN (Kernel-Audit M1): res2 ist hier IDENTISCH NULL, und das ist RICHTIG -- der Zweig erfuellt Gl. 2 exakt, also ist das Gl.-2-Residuum null. Dass QUERGATE fuer ihn nie feuert, ist korrektes Verhalten, kein blinder Fleck. Die Gl.-1-Luecke ist eine ANDERE Groesse (s. LSQ-Zweig).
)+"#endif"+R( // FACETTEN_PINV / FACETTEN_LSQ
	else { if(t%def_zaehl_takt==0ul) atomic_inc(&hits[15]); rueckfall=true; } // Slot 15: gekoppelt Rang 0 (Einzellink c_n!=0) -- BB belassen (Entscheid Gl. 28: jede Erfuellung injizierte Normalimpuls)
	}
	// ★★ 2026-08-25 QUERGATE (CFD_FAC_QUERGATE, Default AUS), Antwort auf Pruefbefund 4-A.
	// Die Skalar-Rueckfaelle treffen ihr Ziel in Stroemungsrichtung exakt und lassen die zweite
	// Gleichung laufen: Phi2 = G12*s1 statt 0, gemessen bis 10^3 mal twe. Das ist derselbe Bruch,
	// den SATGATE fuer die Dynamik schon macht -- iMEM wirkt nur, wenn es sein Ziel erreichen kann.
	// Hier auf die QUERrichtung erweitert: uebersteigt der Restfehler in t2 die Wandschubspannung,
	// die ueberhaupt aufgepraegt werden soll, wird BB belassen statt einen Querimpuls einzuschleppen,
	// den niemand bestellt hat. Ein exakter 2x2- oder Schur-Solve hat res2 = 0 und passiert immer.
)+"#ifdef FACETTEN_QUERGATE"+R(
	if(!rueckfall&&def_fac_tau>0.0f&&res2>def_fac_tau*twe) { if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[64]); if(zweig>0u) atomic_inc(&hits[103u+zweig]); } rueckfall=true; } // Slot 64 -- ★ S2: def_fac_tau>0 als Gate-Vorbedingung, sonst ist der Nullziel-Arm ein Nullschwellen-Gate
)+"#endif"+R( // FACETTEN_QUERGATE
)+"#ifdef FACETTEN_SATGATE"+R(
	// ★ (a-strich), Stabilitaetsanalyse G8: der EINZIGE vorzeichen-definite Injektionsterm ist die
	// GEKLEMMTE Anwendung. Reisst die ungeklemmte Loesung ihr Budget, wird NICHT geklemmt
	// angewandt, sondern BB belassen und gezaehlt -- iMEM wirkt nur, wenn es sein Ziel im Budget
	// exakt erreichen kann (Verallgemeinerung des Rang-0-Entscheids von Geometrie auf Dynamik).
	if(!rueckfall&&((def_fac_isogate>0.5f) ? (s1*s1+s2*s2>4.0f*def_fac_budget*def_fac_budget*ut*ut) : (fabs(s1)>2.0f*def_fac_budget*ut||fabs(s2)>def_fac_budget*ut))) { if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[10]); if(zweig>0u) atomic_inc(&hits[95u+zweig]); } rueckfall=true; } // Slot 10: Gate-Rueckfall; Budget-Skalar def_fac_budget (1a-B4t, Default 1.0 = bitidentisch)
)+"#else"+R(
	const float s1c = clamp(s1, -2.0f*def_fac_budget*ut, 2.0f*def_fac_budget*ut), s2c = clamp(s2, -def_fac_budget*ut, def_fac_budget*ut); // Klemmen (Gl. 9), Budget-Skalar (1a-B4t)
	if(!rueckfall&&(s1c!=s1||s2c!=s2)&&t%def_zaehl_takt==0ul) { atomic_inc(&hits[10]); if(zweig>0u) atomic_inc(&hits[95u+zweig]); } // Slot 10: u_s-Klemme (nicht an Rueckfallzellen zaehlen -- Pruefbefund 4a)
	s1=s1c; s2=s2c;
)+"#endif"+R( // FACETTEN_SATGATE
	// 3x3: sn aus den GEKLEMMTEN s1/s2 (Normal-Nullung haelt auch bei Tangentialklemme, Gl. 23);
	// im entkoppelten Pfad ist sn=0 und Snn ggf. ~0 -- Guard haelt die Division sicher.
	if(!rueckfall&&Snn>=1e-8f&&(Sn1*Sn1+Sn2*Sn2)>1e-6f*Snn*(G11+G22)) {
		sn = -(Sn1*s1+Sn2*s2)/Snn;
)+"#ifdef FACETTEN_SATGATE"+R(
		if(fabs(sn)>def_fac_budget_sn*ut) { if(t%def_zaehl_takt==0ul) { atomic_inc(&hits[16]); if(zweig>0u) atomic_inc(&hits[99u+zweig]); } rueckfall=true; } // Slot 16: sn-Gate-Rueckfall; Budget-Skalar (1a-Bsn)
)+"#else"+R(
		const float snc = clamp(sn, -def_fac_budget_sn*ut, def_fac_budget_sn*ut); // Budget-Skalar (1a-Bsn)
		if(snc!=sn&&t%def_zaehl_takt==0ul) { atomic_inc(&hits[16]); if(zweig>0u) atomic_inc(&hits[99u+zweig]); } // Slot 16: s_n-Klemme
		sn = snc;
)+"#endif"+R( // FACETTEN_SATGATE
	}
	// ★ BUCHUNGSSCHLUSS: an Rueckfallzellen bleibt das FELD reines BB (s=0, Pass 2 uebersprungen), aber die
	// phi-Buchung unten laeuft mit phi = P -> fw = -P_t (+2Dp_t unter ELIBB); zusammen mit der B3-
	// Kopfbuchung -dp ist das exakt der wahre Tangentialaustausch -(2*Sum c_t fpre + d_t) -- die
	// ~40 % Facettenbesuche, die bisher per return NICHTS buchten (K2 -7,4 am 26-Grad-Kanal =
	// Blenden-Korrektur ohne den BB-Anteil, den sie korrigiert). Explizit +0.0f, kein signed-zero-Anker.
)+"#ifdef FACETTEN_MASSE_X"+R(
	if(t%def_zaehl_takt==0ul) { // Kreuztabelle roh x Schatten, dieselbe Stichprobe wie Slot 69 und die Kaskade
		if(a2_rueckfall&&!rueckfall) { if(hits[94]<0xF0000000u) atomic_inc(&hits[94]); if(zweig>0u&&hits[107u+zweig]<0xF0000000u) atomic_inc(&hits[107u+zweig]); } // [94] + [108..111] nach rohem Zweig
		if(rueckfall&&!a2_rueckfall) { if(hits[95]<0xF0000000u) atomic_inc(&hits[95]); if(zweig>0u) { if(hits[111u+zweig]<0xF0000000u) atomic_inc(&hits[111u+zweig]); } else if(hits[116]<0xF0000000u) atomic_inc(&hits[116]); } // [95] + [112..115] nach rohem Zweig, [116] = roh Rang-0
	}
	if(a2_rueckfall) rueckfall=true; // VOR Nullung, pass2_an, kz und beta3
)+"#endif"+R( // FACETTEN_MASSE_X
)+"#ifdef FAC_REK_R3"+R(
	// ★★ R3-GATE. Die Lage ist zwingend: NACH "if(a2_rueckfall) rueckfall=true" (sonst zaehlt die
	// MASSE_X-Kreuztabelle darueber falsch) und VOR der Nullung s1/s2/sn (sonst greift der Rueckfall
	// nicht mehr). KEIN frueher return: seit dem Buchungsschluss vom 27.08. steigen Rueckfallzellen
	// nicht mehr aus, sondern buchen mit s=0 weiter -- phi = P, also fw = -P_t (+2Dp_t unter ELIBB).
	// Ein return hier liesse fac_tau_cnt auf 0, der Kontaminationstest schluege in "nicht
	// kontaminiert" um, und der volle F samt Tangentialanteil wuerde als DRUCK gebucht -- der
	// Zustand, den der Kommentar oben mit K2 = -7,4 am 26-Grad-Kanal beziffert.
	if(t%def_zaehl_takt==0ul) {
		if(rek_gate&&hits[370]<0xF0000000u) atomic_inc(&hits[370]);
		if(rek_gate&&!rueckfall&&hits[331]<0xF0000000u) atomic_inc(&hits[331]);
		if(rek_gate&&rueckfall&&hits[380]<0xF0000000u) atomic_inc(&hits[380]);
	}
	// ★★ DER ZAEHLER, DER DEN DOPPELTERM ENTSCHEIDET (23.09. abends, Pruefbefund H3).
	// P1 wird in der Momentenschleife aus fhn gebildet -- und fhn traegt an den Marken bereits das
	// Df der Rekonstruktion. Unter R3 ist phi1 = P1, also enthaelt die Wandkraft den Wandlink-Anteil
	// des Df EIN ZWEITES MAL, zusaetzlich zur expliziten Buchung fw -= rho*du. Zur fuehrenden
	// Ordnung ist dieser Zusatzterm dP1 = rho*eps*G11roh. Dieselbe Klasse ist bei ELIBB bekannt und
	// wird dort mit +2*Dp_tangential korrigiert.
	// ★ 24.09.2026: DIESER TERM WIRD JETZT KORRIGIERT -- exakt, nicht in fuehrender Ordnung, siehe
	// die Buchung weiter unten. Dieses Histogramm misst seitdem nur noch, WIE GROSS er war; es ist
	// kein offener Befund mehr und keine Handlungsanweisung.
	// G11roh ist das ROHmoment VOR dem ALPHA2-Downdate. Am Fahrzeug laeuft ALPHA2, und "Rang 0
	// NACH dem Downdate" heisst ausdruecklich NICHT G11roh ~ 0: gemessen haben dort nur 8,0 bis 8,4 % der
	// Marken |G11roh| < 1e-6 (24.09., 8-mm-Fahrzeug, alle vier Arme der Serie serie_leiter; die
	// frueher hier genannten 8,3 % waren der Arm l_f8_p4 allein).
	// ★ GEMESSEN 24.09. am kipp26-KANAL: 0,38 % (202 537 von 53 195 217). Der Kanal traegt den
	// Term also staerker als das Fahrzeug -- die frueher hier stehende Entwarnung war falsch.
	// ★ Pruefbefund H2 (24.09.): fuer den kipp26-Kanal gibt es KEINEN Lauf mit diesem Histogramm.
	// Hier stand, der Term sei dort vernachlaessigbar -- das war aus der Restluecke der Serie
	// a2_bilanz nur ERSCHLOSSEN, nicht gemessen. Der Satz ist entfernt, die Messung steht aus.
	if(rek_gate&&t%def_zaehl_takt==0ul) {
		const float g11a = fabs(rek_g11);
		const uint gb = g11a<1.0E-6f ? 0u : (g11a<1.0E-4f ? 1u : (g11a<1.0E-2f ? 2u : (g11a<1.0f ? 3u : 4u)));
		if(hits[373ul+(ulong)gb]<0xF0000000u) atomic_inc(&hits[373ul+(ulong)gb]);
	}
	if(rek_gate) rueckfall=true;
)+"#endif"+R( // FAC_REK_R3
	if(rueckfall) { s1=0.0f; s2=0.0f; sn=0.0f; }
	// ★★ ZELLKRAFT STATT SLIP (CFD_FAC_KRAFT, 30.08.2026, Planungsagent Weg F -- IVW-Hybrid nach
	// Kuwata & Suga). Befund: das Gate feuert, wenn |P1| > 2*G11*ut -- eine Eigenschaft der
	// LINKMENGE (Kopplung Sn1/G11, Schur-Verstaerkung), nicht der Physik; im Nullziel-Arm feuert es
	// HAEUFIGER als mit Ziel. Klemme (SATGATE=0) hat einen vorzeichen-definiten Bias (G8): 8 mm
	// cz_druck_rest -0,152 -> -0,050, 4 mm -1,016 -> -0,759. Hier stattdessen: das Residuum R, das
	// der Solve in span(L) nicht erreicht, als VOLUMENKRAFT in R^3 -- massenexakt, ohne Rang-,
	// Budget- oder Positivitaetsfrage, Normalanteil per Konstruktion 0. Modus 1: nur Rueckfallzellen
	// (kipp0 hat keine -> bitgleich). Modus 2: ALLE Facettenzellen per Kraft, Additivterm aus --
	// der Diskriminator gegen den Slip-Pfad (0,716 an der ebenen Wand).
)+"#else"+R(
	// ★★★ 06.09.2026 KINEMATISCHES WANDMODELL (CFD_FAC_UW, Ponsin & Lozano 2025).
	// Statt ein Gleichungssystem zu loesen wird die Wandgeschwindigkeit HINGESCHRIEBEN:
	//   u_w = u_B - u_tau^2 * delta_w / (nu + nu_t)
	// so dass der aufgeloeste Gradient zwischen Wand und Abtastpunkt genau tau_w = rho*u_tau^2 traegt.
	// WARUM DAS DIE NICHT-ACHSPARALLELEN ZELLEN ERREICHT: die Sperre an Ein-Link-Facetten war
	// J || c zusammen mit der Forderung J.n = 0, woraus J = 0 folgt. Diese Forderung gehoerte zum
	// SOLVE. Hier gibt es keinen Solve mehr, also auch keine Nebenbedingung -- q_i = 6 w_i (c_i.u_w)
	// ist an einem einzigen Link von null verschieden. Genau so haelt es der Bewegtwand-Zwilling
	// apply_moving_boundaries (Krueger S. 180), der ebenfalls kein J.n = 0 fordert.
	// ★★★ WARNUNG 06.09.2026 abends, NACH DER AUDIT-SCHLEIFE: DIESER ARM IST WIDERLEGT.
	// NICHT BENUTZEN, bevor die Konvention repariert ist. Drei unabhaengige Auditoren, uebereinstimmend:
	//  (1) Der diskrete Bounce-Back liefert je Schritt den Tangentialfluss G11*(u_B - u_w) = (1/3)*(u_B-u_w),
	//      NICHT rho*nu_eff*(u_B-u_w)/y_w. Fuer tau_w = rho*u_tau^2 muesste (u_B - u_w) = 3*u_tau^2 sein;
	//      hier steht u_tau^2*y_w/nu_eff. Verhaeltnis y_w/(3*nu_eff) ~ 269. Kontinuumskonvention in einem
	//      Platz, dessen Wert die Gitterkonvention bestimmt.
	//  (2) s1 ist KEINE kinematische Wandgeschwindigkeit, sondern der Koeffizient, der die
	//      BB-Populationsunwucht P1 wegheben muss (|P1|/twe = 3760). s1_noetig = (twe+|P1|)/G11 = 1,29*u_B
	//      -- also OBERHALB von u_B. Die Klemme unten schneidet genau den einzigen zulaessigen Bereich ab.
	//      An der Kugel brauchen 54,2 % der Facetten einen Wert ausserhalb [0, u_B].
	//  (3) Die eigene Projektdoku hatte es ausgeschlossen: WANDMODELL.md sagt "Abtastung 2 Zellen von der
	//      Wand" und "Die Ankopplung ueber eine effektive Viskositaet SCHEIDET AUS (Ponsin & Lozano 2025)".
	//      Gebaut wurde Abtastung IN der Wandzelle und Ankopplung UEBER eine effektive Viskositaet.
	// Gemessen: Kanal u_tau IST/Ziel 0,816 -> 7,2, Zielerfuellung r = -1281 (Gegenrichtung, 96,2 % der
	// Besuche); Kugel r <= -10 bei 73,7 %, Normal-Rest x 2,3e5. Die Ein-Link-Klasse, fuer die der Umbau
	// gemacht wurde, war als reiner BB-Rueckfall bei r = 0,88 und steht mit u_w bei r = -1,87 --
	// die neue Behandlung ist schlechter als gar keine.
	// SCOPE-HINWEIS: utau aus der Spalding-Kette lebt in deren eigenem Block (oben) und ist hier NICHT
	// sichtbar; tw dagegen schon. tw = rhon*utau^2 ist die Definition, also ist die Wurzel exakt
	// dasselbe u_tau -- inklusive der Stabilitaetsklemme tw_max, was gewollt ist. NICHT twe nehmen:
	// darin steckt der Flaechenfaktor faca, und u_w ist Kinematik, keine Flaechenbilanz.
	{	const float utau_uw = sqrt(fmax(tw, 0.0f)/fmax(rhon, 1e-6f));
		const float yp_uw = utau_uw*yw_ab/def_fac_nu;                 // y+ am Abtastpunkt
		// MODUS 1 (der einzige gebaute): nu_t aus dem GLEICHGEWICHTSPROFIL, nu_eff/nu = 1 + kappa*y+.
		// Modus 2 -- nu_t aus dem gemessenen Feld (fac_wfd, Kernel sgs_fdwand kernel.cpp:4281ff) -- ist
		// bewusst NICHT gebaut: er koppelt an CFD_SGS_FDWAND und waere eine zweite Variable im selben
		// Schritt. Er braucht ausserdem ein neues Kernelargument (fac_wfd ist hier kein Parameter) und
		// ist im Kugelfall gar nicht verdrahtet (setup.cpp: FDWAND dort nicht gesetzt). Eigener Schritt.
		const float rnu_uw = 1.0f + def_fac_uwkappa*yp_uw;
		// KLEMME, zwingend: bei nu_t -> 0 (SUBGRID aus, SGS_WANDFREI) verlangt die Formel eine
		// rueckwaerts laufende Wand -- bei y+ = 100 waere u_w = -5*u_B. Die untere Klemme u_w = 0 ist
		// bitgenau reines Bounce-Back (q_i = 6 w_i (c_i.0) = 0), Klemme und Rueckfall fallen zusammen.
		// ★ BERICHTIGT 06.09.: die obere Klemme ist KEIN Nullbeweis. duw >= 0 gilt immer, also ist
		// uw > ut_ab konstruktiv unmoeglich und Slot 125 kann nie feuern -- ein Test, der nicht
		// fehlschlagen kann, ist keiner. Schlimmer: die RICHTIGE Loesung liegt oberhalb u_B (1,29*u_B),
		// diese Klemme schneidet also genau den zulaessigen Bereich ab.
		const float duw = utau_uw*utau_uw*yw_ab/(def_fac_nu*rnu_uw);
		float uw = ut_ab - duw;
		if(uw<=0.0f) { uw = 0.0f; if(t%def_zaehl_takt==0ul&&hits[124]<0xF0000000u) atomic_inc(&hits[124]); }
		if(uw>ut_ab) { uw = ut_ab; if(t%def_zaehl_takt==0ul&&hits[125]<0xF0000000u) atomic_inc(&hits[125]); }
		s1 = uw; s2 = 0.0f; sn = 0.0f;
		rueckfall = (uw<=0.0f); // nur die Klemme faellt zurueck -- Rang und Gates gibt es hier nicht
		if(t%def_zaehl_takt==0ul&&hits[123]<0xF0000000u) atomic_inc(&hits[123]); // Wirkpfad: MUSS feuern
		{	const float q_uw = (ut_ab>1e-12f) ? uw/ut_ab : 0.0f;         // Histogramm u_w/u_B, 8 Eimer a 0,125
			const uint b_uw = (uint)fmin(7.0f, fmax(0.0f, floor(8.0f*q_uw)));
			if(t%def_zaehl_takt==0ul&&hits[128u+b_uw]<0xF0000000u) atomic_inc(&hits[128u+b_uw]); }
)+"#ifdef FACETTEN_UW_SN"+R(
		// A/B-Arm: Normalnullung trotz u_w. Dann faellt die Ein-Link-Klasse wie heute zurueck, und die
		// Differenz der beiden Arme MISST den Preis der Nebenbedingung J.n = 0.
		if(!rueckfall&&Snn>=1e-8f) sn = -(Sn1*s1)/Snn;
)+"#endif"+R( // FACETTEN_UW_SN
	}
)+"#endif"+R( // FACETTEN_UW
	bool pass2_an = !rueckfall;
	float3 kraft = (float3)(0.0f,0.0f,0.0f);
)+"#ifndef FACETTEN_UW"+R(
)+"#ifdef FACETTEN_KRAFT"+R(
	const bool kz = rueckfall || (def_fac_kraft==2u);
	if(kz) {
		kraft = (float3)(R1*t1x+R2*t2x, R1*t1y+R2*t2y, R1*t1z+R2*t2z); // = Ziel - P, tangential
		s1=0.0f; s2=0.0f; sn=0.0f; pass2_an=false;                    // kein Additivterm an Kraftzellen
		if(t%def_zaehl_takt==0ul&&hits[70]<0xF0000000u) atomic_inc(&hits[70]); // Slot 70: Kraftpfad (saettigend)
		if(t<100ul&&hits[71]<0xF0000000u) atomic_inc(&hits[71]); // Slot 71: Kraftzellen im ANLAUF (t<100), UNGEGATET -- Bitanker-Befund 30.08.: kipp0 hat am Startschritt an ALLEN Facettenzellen Rueckfall (3720 = fac_N), die t%100-Stichprobe sieht das nicht
	}
)+"#endif"+R( // FACETTEN_KRAFT
)+"#endif"+R( // FACETTEN_UW -- KRAFT braucht R1/R2, die es unter u_w nicht gibt
	float usx = s1*t1x+s2*t2x+sn*nx, usy = s1*t1y+s2*t2y+sn*ny, usz = s1*t1z+s2*t2z+sn*nz;
)+"#ifdef FAC_REK_R3"+R(
	// ★★ BAUREIHENFOLGE-PROBE (NICHT der Nullbeweis des Gates -- berichtigt 23.09. abends nach
	// Pruefbefund B7/M3). Unter der ERLAUBTEN Schalterschnittmenge ist usx/usy/usz an einer Marke
	// konstruktiv 0, weil zwischen Gate und hier nur genullt wird; die beiden Wege zu usx != 0
	// (FACETTEN_UW, FACETTEN_KRAFT) sind beide hart gesperrt. Dieser Zaehler feuert also nur, wenn
	// das Gate hinter die Nullung wandert, ein Filter u_s neu belegt oder t1 nicht endlich ist --
	// ein REGRESSIONSWAECHTER. Den Nullbeweis des Tors liefert allein der Hashvergleich Arm 1
	// gegen Arm 2. VERSETZT am 23.09. (Pruefbefund H4). Er stand zuerst direkt hinter
	// "pass2_an = !rueckfall" -- dort war er TAUTOLOGISCH: zwischen dem Gate und dieser Zeile wird
	// rueckfall nur im #else-Zweig von #ifndef FACETTEN_UW neu gesetzt, und in dem wird das Gate gar
	// nicht emittiert. Der Zaehler konnte konstruktiv nie feuern, und der Host druckte seine Null als
	// Beleg. Genau die Klasse, fuer die Slot 331 heute frueh von drei Auditoren entfernt wurde.
	// HIER prueft er etwas Echtes: u_s ist der Additivterm, der auf die Zelle geht. An einer Marke
	// MUSS er null sein -- sonst steht das Gate hinter der Nullung, oder ein Filter (EMA/PEMA) hat
	// Soll EXAKT 0. (Der zuerst hier genannte Filterfall EMA/PEMA traegt NICHT: der EMA-Block steht
	// hinter diesem Zaehler -- Nachpruefung NEU-4. Die drei anderen Nachweise sind der Zweck.)
	if(rek_gate&&(usx!=0.0f||usy!=0.0f||usz!=0.0f)&&t%def_zaehl_takt==0ul&&hits[371]<0xF0000000u) atomic_inc(&hits[371]);
)+"#endif"+R( // FAC_REK_R3
)+"#ifdef FACETTEN_EMA"+R(
	// LATENT (Audit 1/3): unter EMA x SATGATE prueft das Gate die GELOESTEN s, angewandt wird die
	// EMA-Mischung -- die bei fallendem ut das Budget ueberschreiten kann. EMA ist widerlegter
	// Legacy-Arm; wer ihn je reaktiviert, muss das Gate hinter den Filter ziehen.
	// ★ Asmuth Gl. 29/30 / A6-Plan: EMA von u_s im xyz-Rahmen (frame-stabil) -- das instantane
	// Nullziel jagte P-Fluktuationen und pumpte mit +-2ut-Oszillation selbst Turbulenz (J0-Befund
	// 45 Grad: Klemmquote 8 %, Ist-Reibung 84 % der Wandreibung trotz Nullziel).
	if(!rueckfall) { // Legacy-EMA: Rueckfall verliess frueher vor dem Filter
		const uxx e = 3ul*(uxx)fid;
		const float ax = def_fac_ema;
		usx = fma(ax, usx-fac_us[e], fac_us[e]); usy = fma(ax, usy-fac_us[e+1ul], fac_us[e+1ul]); usz = fma(ax, usz-fac_us[e+2ul], fac_us[e+2ul]);
		fac_us[e]=usx; fac_us[e+1ul]=usy; fac_us[e+2ul]=usz;
	}
	// Projektionen des ANGEWANDTEN u_s fuer Ist-Kraft/Delta-m/Rest (nur im EMA-Arm noetig)
	s1 = usx*t1x+usy*t1y+usz*t1z; s2 = usx*t2x+usy*t2y+usz*t2z; sn = usx*nx+usy*ny+usz*nz;
)+"#endif"+R( // FACETTEN_EMA
)+"#ifdef FACETTEN_ALPHA"+R(
	// ★ J4-alpha (Massenkorrektur): Sum_i 6 w_i (c_i*u_s) = 6(S1*u_s) != 0 bei unsymmetrischer
	// Linkmenge -- jede Facette injizierte netto Masse (Kugel Delta-m~269, arm-unabhaengig, am
	// Druck-Bookkeeping vorbei). alpha aus dem ANGEWANDTEN u_s (nach Gate/Klemme/EMA), damit
	// Sum q = alpha*S0 + 6(S1*u_s) = 0 exakt gilt, egal was Gates und Filter getan haben.
	// S0 >= w(1) > 0 hier: ohne Wandlink waere der Solver oben mit Rang 0 ausgestiegen.
	const float alph = (S0>0.0f) ? -6.0f*(S1x*usx+S1y*usy+S1z*usz)/S0 : 0.0f; // Auditor A Befund 2 (07.09.): Nullschutz. Die alte Begruendung (S0 >= w(1) > 0, sonst waere der Solver mit Rang 0 ausgestiegen) ist seit dem Buchungsschluss vom 27.08. hinfaellig -- Rang 0 steigt nicht mehr per return aus, sondern laeuft mit rueckfall=true weiter. Eine aktive Facette ohne einen einzigen Wandlink haette 0/0 = NaN geliefert; unter ALPHA=1 haette das ueber phi1 den GANZEN Reibungspfad vergiftet. Bitgleich fuer jeden Lauf mit S0 > 0.
)+"#ifdef FACETTEN_MASSE_ALLE"+R(
	// ★ 04.09. (Diff-Pruefung d6): unter MASSE_ALLE wird beta3 = alph*S0 injiziert, nicht alph. Mit
	// S0 ~ 0,1..0,3 warnte der Waechter auf einem drei- bis zehnfach zu grossen Wert.
	if(fabs(-6.0f*(S1x*usx+S1y*usy+S1z*usz))>ut&&t%def_zaehl_takt==0ul) atomic_inc(&hits[18]);
)+"#else"+R(
	if(fabs(alph)>ut&&t%def_zaehl_takt==0ul) atomic_inc(&hits[18]); // Slot 18: alpha in Geschwindigkeitsordnung -- Warnsignal
)+"#endif"+R(
)+"#ifdef FACETTEN_MASSE_ALLE"+R(
	// ★★ 04.09.2026 STUFE 3: DIESELBE Masse, aber ueber ALLE 19 Links statt nur ueber die Wandlinks.
	// Der Grund ist der gemessene Preis von Stufe 2: das Downdate kostet 36,26 % der Facetten eine
	// Rangstufe (statischer Zensus, 8-mm-Fahrzeug: 240.966 Facetten fallen von Rang 2 auf Rang 1,
	// 19.969 von Rang 1 auf Rang 0). Und das Downdate ist kein Zusatz, sondern die ehrliche
	// Beschreibung dessen, was die Wandlink-Verteilung anrichtet:
	//   Stufe 1/2: Sum_Wand w_i alph c_i = alph*S1 = -(6/S0)(S1.u_s) S1   -> genau der Downdate-Term
	//   Stufe 3:   Sum_alle w_i beta c_i = beta * Sum w_i c_i = 0          -> KEIN Impuls
	// Massenprobe, exakt und ZELLWEISE (nicht global): Sum q + Sum p = 6(S1.u_s) + beta*Sum w_i
	// = 6(S1.u_s) - 6(S1.u_s) = 0, weil fuer D3Q19 Sum w_i = 1 gilt (1/3 + 6/18 + 12/36).
	// Der angewandte Operator ist damit die ROHE Momentenmatrix -- voller Rang, ohne Masseleck.
	const float beta3 = -6.0f*(S1x*usx+S1y*usy+S1z*usz);
)+"#endif"+R( // FACETTEN_MASSE_ALLE
)+"#endif"+R( // FACETTEN_ALPHA
	if(pass2_an) for(uint i=1u; i<def_velocity_set; i++) { // Pass 2 (pass2_an == !rueckfall ohne KRAFT): q_i = 6 w_i (c_i*u_s) addieren (Gl. 3); Rueckfall: Feld bleibt BITGLEICH BB
		const uint ib = (i%2u==1u) ? i+1u : i-1u;
		if((flags[j[ib]]&TYPE_BO)!=TYPE_S) continue;
		fhn[i] = fma(6.0f*w(i), c(i)*usx+c(def_velocity_set+i)*usy+c(2u*def_velocity_set+i)*usz, fhn[i]);
)+"#ifdef FACETTEN_ALPHA"+R(
)+"#ifndef FACETTEN_MASSE_ALLE"+R(
		fhn[i] += w(i)*alph; // Stufe 1/2: nur ueber die Wandlinks -- genau das erzeugt den Downdate-Term
)+"#endif"+R( // !FACETTEN_MASSE_ALLE
)+"#endif"+R( // FACETTEN_ALPHA
	}
)+"#ifdef FACETTEN_MASSE_ALLE"+R(
	// Ungegatet ueber ALLE Richtungen inklusive i=0 -- Sum w_i = 1 traegt die Masse, Sum w_i c_i = 0
	// sorgt dafuer, dass dabei kein Impuls entsteht. fhn ist das volle lokale 19er-Feld und geht
	// vollstaendig durch store_f, fhn[0] eingeschlossen.
	float masse_ist = 0.0f; // was WIRKLICH verteilt wurde -- Grundlage der Delta-m-Buchung unten
	if(pass2_an) {
)+"#if defined(FACETTEN_MASSE_F0)"+R(
		// ★★ MODUS 2 (04.09.2026, Diff-Pruefung c): die gesamte Kompensation auf die RUHEPOPULATION.
		// Weil c_0 = 0 ist, traegt sie Masse, aber WEDER Impuls NOCH zweiten Moment -- sie ist die
		// einzige Verteilung, die keinen anderen Moment anfasst.
		// ★ KORRIGIERT 04.09. abends (Planungsagent B1): die Rollen waren hier VERTAUSCHT. Modus 1
		// addiert p_i = w_i*beta3 = f_eq(rho=beta3, u=0) -- sein zweiter Moment beta3/3*I ist der
		// GLEICHGEWICHTSDRUCK der zugefuegten Masse, kein Spannungsterm; sein Nichtgleichgewichtsanteil
		// ist O(beta3*u^2), rund 1e-3 des Slip-Terms. DIESER Modus 2 dagegen traegt relativ zum
		// Gleichgewicht Pi_neq = -beta3/3*I: ein isotroper Nichtgleichgewichts-Bulk-Mode an ~70 % der
		// Wandzellen je Schritt, der bei omega ~ 2 ueberrelaxiert. GEMESSEN am 8-mm-Fahrzeug (vo_f08_m2):
		// Geschwindigkeitsklemme 14.659.833 gegen 1.570 in der Basis (x9.300), f_0 <= 0 bei 2,0 % der
		// Besuche (Slot 93), cz_druck_rest +0,675. Modus 2 ist VERWORFEN; der Zaehler bleibt als Beleg.
		fhn[0] += beta3; masse_ist = beta3; // D15: masse_ist == beta3 EXAKT -> die Delta-m-Buchung ist hier konstruktiv 0 (kein Messwert; Modus verworfen 04.09.)
		if(fhn[0]+0.33333334f<=0.0f&&t%def_zaehl_takt==0ul&&hits[93]<0xF0000000u) atomic_inc(&hits[93]); // [93] f_0 nicht mehr positiv
)+"#else"+R(
		for(uint i=0u; i<def_velocity_set; i++) { fhn[i] += w(i)*beta3; masse_ist += w(i)*beta3; }
)+"#endif"+R(
		if(t%def_zaehl_takt==0ul&&hits[92]<0xF0000000u) atomic_inc(&hits[92]); } // [92] Wirkpfad
)+"#endif"+R( // FACETTEN_MASSE_ALLE
	float phi1 = P1 + fma(G11,s1,G12*s2) + Sn1*sn, phi2 = P2 + fma(G12,s1,G22*s2) + Sn2*sn;
)+"#ifndef FACETTEN_UW"+R(
)+"#ifdef FACETTEN_KRAFT"+R(
	if(kz) { phi1 += R1; phi2 += R2; } // Kraftzelle: Ist = P + Kraft = Ziel (Buchung Ist == Soll, wie im Slip-Pfad)
)+"#endif"+R( // FACETTEN_KRAFT // Ist-Austausch nach Klemme (3x3: inkl. Sn-Beitrag des sn; unter ALPHA2 sind G/Sn downgedatet -> alpha-Beitrag enthalten)
)+"#endif"+R( // FACETTEN_UW -- kz/R1/R2 existieren unter u_w nicht (Pruefbefund H1)
)+"#ifdef FACETTEN_ALPHA"+R(
)+"#ifndef FACETTEN_ALPHA2"+R(
)+"#ifndef FACETTEN_MASSE_ALLE"+R(
	// Stufe 1 traegt den alpha-Impuls (alpha*S1) NICHT im Downdate -- fuer die ehrliche Ist-Kraft addieren:
	// Unter Stufe 3 entfaellt das: die Kompensation ueber alle Links traegt konstruktiv KEINEN Impuls,
	// und G ist roh, also ist phi1 = P1 + G_roh*s bereits das vollstaendige Ist.
	phi1 += alph*(S1x*t1x+S1y*t1y+S1z*t1z); phi2 += alph*(S1x*t2x+S1y*t2y+S1z*t2z);
)+"#endif"+R( // !FACETTEN_MASSE_ALLE
)+"#endif"+R(
)+"#endif"+R( // FACETTEN_ALPHA
	// ★ 04.09.2026 ZIELERFUELLUNG (Slots 81-91). Der Anlass: "Anteil der Wandbehandlungen MIT Modell"
	// sagt, wieviele Besuche ein Modell BEKOMMEN haben -- nicht, wieviel vom Wandschub-Ziel dabei
	// ankommt. Die Rang-1-Pseudoinverse erfuellt R nur in Richtung ihres einen erreichbaren
	// Eigenvektors; der Rest bleibt liegen und wurde bisher von NICHTS gemessen (res2 sieht nur t2).
	// Gemessen wird r = phi1 / (-def_fac_tau*twe): phi1 ist der tatsaechlich aufgepraegte
	// Tangentialimpuls (Pass 2 + alpha, unter ALPHA2 sind G/Sn downgedatet, also enthalten), das Ziel
	// ist -def_fac_tau*twe. Exakte Erfuellung => r == 1.
	// WARUM JE BESUCH UND NICHT ALS SUMME (Vorpruefagent 04.09., an export/zh_pinv4 nachgerechnet):
	// phi1 traegt P1 mit, und |P1|/twe liegt am Fahrzeug je Klasse zwischen 25 und 1145. Der Solver
	// hebt zu ueber 99 % P1 weg. Sum(Ist)/Sum(Soll) ist damit der Quotient zweier winziger Reste
	// zweier fast gleicher Riesen -- an der 4-mm-Rechnung -12,07 statt 1, und 369 von 440 Klassen
	// mit umgekehrtem Vorzeichen. Das Verhaeltnis JE BESUCH normiert sich selbst.
	// Stichprobe wie beim Momenten-Histogramm 2129: t%100 UND jede 64. Facette (uint-Wickel).
	{	const float zi_ze = def_fac_tau*twe;
		if(t%def_zaehl_takt==0ul&&((fid*2654435761u)&4227858432u)==0u) {
			if(hits[81]<0xF0000000u) atomic_inc(&hits[81]); // [81] Nenner: alle gestichprobten Besuche
			if(pass2_an) {
				if(zi_ze>0.0f) {
					const float r_ze = phi1/(-zi_ze);
					// ★ 04.09. NACHGESCHAERFT nach dem Erstlauf (vc_ziel, Kanal kipp26): der erste Entwurf
					// hatte EINEN Eimer fuer r<0 und der lief mit 57,3 % voll. Das wirft "leicht in die
					// Gegenrichtung" mit "um Dekaden daneben" zusammen -- bei |P1|/Ziel zwischen 25 und
					// 1145 (facetten_klassen.csv) ist genau das der diagnostisch entscheidende Unterschied:
					// ein nicht weggehobenes P1 erscheint als GROSSER Betrag, eine echte Vorzeichenumkehr
					// des Wandschubs als kleiner.
					const uint b_ze = (r_ze<=-10.0f)?0u:((r_ze<=-1.0f)?1u:((r_ze<0.0f)?2u:((r_ze<0.5f)?3u:((r_ze<0.9f)?4u:((r_ze<1.1f)?5u:((r_ze<2.0f)?6u:((r_ze<10.0f)?7u:8u)))))));
					if(hits[83u+b_ze]<0xF0000000u) atomic_inc(&hits[83u+b_ze]); // [83..91]
					// ★ 05.09. REST-DRUCKTERM (Slots 118..122), HIERHER VERLEGT nach Pruefbefund H-4.
					// Er stand zuerst im Stoerform-Messblock (~2130), und der ist NICHT auf pass2_an gegatet:
					// am Kanal kipp26 sind nur 26,82 % der gestichprobten Besuche ueberhaupt angewandt, an den
					// uebrigen 73 % laeuft reines Bounce-Back mit s=0 und der Loeser hebt GAR NICHTS weg.
					// "In 100 % der Besuche" umfasste damit zu drei Vierteln Zellen, an denen der behauptete
					// Mechanismus konstruktiv nicht stattfindet. Hier steht er im selben if-Rumpf wie [83..91]
					// und teilt deren Guard-Trippel (Stichprobe, pass2_an, zi_ze>0) KONSTRUKTIV.
					// GEMESSEN WIRD: |2*(rho-1)*(S1.t1)| / |def_fac_tau*twe| -- der isotrope Druckanteil, der
					// nach der Stoerform (fhn = f - w_i) in P1 VERBLEIBT und damit ungemessen im Ziel steht.
					// Nur die t1-Komponente: in R1 geht ausschliesslich S1.t1 ein, S1.t2 gehoert zu R2 (Ziel 0).
					// ABNAHME: Summe(118..122) == Summe(83..91), exakt, in jedem Arm.
					{	const float b1_ze = S1x*t1x+S1y*t1y+S1z*t1z;
						const float rr_ze = fabs(rhon-1.0f)*fabs(2.0f*b1_ze)/zi_ze;
						const uint b_rr = rr_ze<0.001f?0u:(rr_ze<0.01f?1u:(rr_ze<0.1f?2u:(rr_ze<1.0f?3u:4u)));
						if(hits[118u+b_rr]<0xF0000000u) atomic_inc(&hits[118u+b_rr]); }
				} else if(hits[82]<0xF0000000u) atomic_inc(&hits[82]); // [82] angewandt, aber kein Ziel (twe==0)
			}
		}
	}
	float fwx = -(phi1*t1x+phi2*t2x), fwy = -(phi1*t1y+phi2*t2y), fwz = -(phi1*t1z+phi2*t2z);
)+"#ifdef FACETTEN_ELIBB"+R(
	// ★★ B3-Pruefbefund 1b (2026-08-25 nacht, HART): P1 wird NACH der Blende gemessen -- das
	// MEM-2x von P1 setzt f_out = f_in voraus, die Blende bricht das. Ihr Tangentialanteil steckt
	// damit bereits MIT FAKTOR 2 in der phi-Buchung; die B3-Kopfbuchung (-Dp) machte die Summe
	// tangential vorzeichenverkehrt. Korrektur (numerisch verifiziert, schliesst auf ~1e-18):
	// +2*Dp_tangential HIER. Der Normalanteil der Kopfbuchung bleibt die korrekte Einfachzaehlung;
	// am Pur-Pfad (phi bucht nichts) ist die Kopfbuchung allein exakt; an RUECKFALLZELLEN bucht phi
	// seit dem Buchungsschluss (27.08.) P-only + 2Dp_t -- zusammen mit -dp exakt -(2 Sum c_t fpre + d_t).
	{
		const float edn_ = elibb_dp.x*nx+elibb_dp.y*ny+elibb_dp.z*nz;
		fwx += 2.0f*(elibb_dp.x-edn_*nx); fwy += 2.0f*(elibb_dp.y-edn_*ny); fwz += 2.0f*(elibb_dp.z-edn_*nz);
	}
)+"#endif"+R( // FACETTEN_ELIBB
)+"#ifdef FAC_R1Q"+R(
	// ★★ 28.09.2026 RANG-1-QUERREST (CFD_FAC_R1Q). Hier stehen P, Pass 2, phi und fw fest; fhn wird danach
	// in dieser Funktion nicht mehr gelesen, und die Kollision laeuft im SELBEN Schritt auf u+du.
	// Ziel Z = (Z1, 0) in (t1, t2), Z1 = -def_fac_tau*twe. Der PINV-Zweig praegt G~*s_Z = Z1*(a^2+b^2, b) auf
	// (a = Gt11/tr, b = Gt12/tr, Gt22/tr = 1-a). Es fehlt m = (Z1*(1-a^2-b^2), -Z1*b): die t1-Komponente ist
	// der fehlende Wandschub in Stroemungsrichtung (Anteil 1-cos^2), die t2-Komponente nimmt den Querimpuls
	// Z1*cos*sin zurueck, den die PINV quer zur Stroemung einpraegt. rho*du = m, masselos, tangential.
	// NUR an Marken (Lage 1, statischer Rang 1) UND nur, wenn in diesem Besuch der PINV-Zweig angewandt wurde.
	{
		const bool r1q_zs = (t%def_zaehl_takt==0ul);
		const bool r1q_marke = (fac_geo[b+7ul] < -0.5f);
		const bool r1q_quelle = r1q_pinv&&pass2_an;
		if(r1q_zs) {
			if(r1q_marke) {
				if(hits[408]<0xF0000000u) atomic_inc(&hits[408]);
				const uint r1q_fach = r1q_quelle ? 409u : (r1q_pinv ? 410u : (zweig==3u ? 411u : ((zweig==1u||zweig==2u) ? 412u : 413u)));
				if(hits[r1q_fach]<0xF0000000u) atomic_inc(&hits[r1q_fach]);
			}
			else if(r1q_quelle&&hits[414]<0xF0000000u) atomic_inc(&hits[414]);
		}
		if(r1q_marke&&r1q_quelle) {
			const float r1q_z1 = -def_fac_tau*twe;
			const float r1q_q = 1.0f-r1q_a*r1q_a-r1q_b*r1q_b;
)+"#ifdef FAC_R1Q_VR"+R(
			// ★ 28.09. V_R (CFD_FAC_R1Q=3): der VOLLE Rest (I-M)R mit R = Z - P, M = Gt/tr = [[a,b],[b,1-a]].
			// Nach dem PINV-Solve ist phi = P + M*R; ein Rang-2-Solve haette phi = Z. Es fehlt (I-M)R: quer zur
			// loesbaren Richtung wird der Bounce-Back-Austausch durch das Wandmodellziel ERSETZT -- genau das, was der
			// Solve an Rang-2-Zellen ohnehin tut. V_Z (nur (I-M)Z) war am Fahrzeug wirkungslos (r1q_f8_vz, 28.09.).
)+"#ifdef FAC_R1Q_OHNE_DRUCK"+R(
			// ★ 28.09. R1Q=4: V_R OHNE den isotropen Druckanteil des BB-Austauschs. P enthaelt A = 2(rho-1)(S1.t)
			// (derselbe Term wie KDIAG [12], S1 ROH); ersetzt wird nur der Reibungsanteil P - A, also R' = R + A.
			// Anlass: V_R (R1Q=3) am Fahrzeug -- Grenzschicht vorn OF13-duenn, aber Abloesung 0,4 m frueher (r1q_f8_vr).
			const float r1q_ad = 2.0f*(rhon-1.0f);
			const float r1q_r1 = fma(r1q_ad, S1x*t1x+S1y*t1y+S1z*t1z, R1);
			const float r1q_r2 = fma(r1q_ad, S1x*t2x+S1y*t2y+S1z*t2z, R2);
)+"#else"+R(
			const float r1q_r1 = R1;
			const float r1q_r2 = R2;
)+"#endif"+R( // FAC_R1Q_OHNE_DRUCK
			const float r1q_m1 = (1.0f-r1q_a)*r1q_r1-r1q_b*r1q_r2;
			const float r1q_m2 = r1q_a*r1q_r2-r1q_b*r1q_r1;
)+"#else"+R(
			const float r1q_m1 = r1q_z1*r1q_q;
			const float r1q_m2 = -r1q_z1*r1q_b;
)+"#endif"+R( // FAC_R1Q_VR
			const float r1q_ir = 1.0f/rhon;
			const float r1q_dux = (r1q_m1*t1x+r1q_m2*t2x)*r1q_ir;
			const float r1q_duy = (r1q_m1*t1y+r1q_m2*t2y)*r1q_ir;
			const float r1q_duz = (r1q_m1*t1z+r1q_m2*t2z)*r1q_ir;
			const float r1q_dl = sqrt(fma(r1q_dux, r1q_dux, fma(r1q_duy, r1q_duy, r1q_duz*r1q_duz)));
			if(r1q_zs) {
				const uint r1q_bq = r1q_q<0.1f ? 0u : (r1q_q<0.3f ? 1u : (r1q_q<0.5f ? 2u : (r1q_q<0.7f ? 3u : (r1q_q<0.9f ? 4u : 5u))));
				if(hits[415u+r1q_bq]<0xF0000000u) atomic_inc(&hits[415u+r1q_bq]);
				const float r1q_pp1 = P1-(r1q_a*P1+r1q_b*P2);
				const float r1q_pp2 = r1q_a*P2-r1q_b*P1;
				const float r1q_pr = sqrt(fma(r1q_pp1, r1q_pp1, r1q_pp2*r1q_pp2))/fmax(fabs(r1q_z1), 1.0E-30f);
				const uint r1q_bp = (r1q_z1==0.0f) ? 5u : (r1q_pr<0.1f ? 0u : (r1q_pr<1.0f ? 1u : (r1q_pr<10.0f ? 2u : (r1q_pr<100.0f ? 3u : 4u))));
				if(hits[421u+r1q_bp]<0xF0000000u) atomic_inc(&hits[421u+r1q_bp]);
				if(r1q_dl>0.5f*ut*1.0001f&&hits[427]<0xF0000000u) atomic_inc(&hits[427]);
				// ★ 28.09.: unter V_Z ist 427 ein Stolperdraht (Soll 0), unter V_R das TOR -- dieselbe Schranke wie tw_max = 0,5*rho*ut.
				if(fabs(r1q_dux*nx+r1q_duy*ny+r1q_duz*nz)>1.0E-3f*r1q_dl+1.0E-12f&&hits[431]<0xF0000000u) atomic_inc(&hits[431]);
			}
)+"#ifdef FAC_R1Q_AN"+R(
			if(!(r1q_dl>0.5f*ut*1.0001f)) {
			const float r1q_fw1 = fwx*t1x+fwy*t1y+fwz*t1z;
			const float r1q_fw2 = fwx*t2x+fwy*t2y+fwz*t2z;
			const float r1q_fwb = fabs(fwx)+fabs(fwy)+fabs(fwz);
			const float4 r1q_inj = r1q_einspeisen(fhn, rhon, uxn, uyn, uzn, r1q_dux, r1q_duy, r1q_duz);
			fwx -= rhon*r1q_dux;
			fwy -= rhon*r1q_duy;
			fwz -= rhon*r1q_duz;
			if(r1q_zs) {
				const float r1q_mb = rhon*r1q_dl;
				const float r1q_ex = r1q_inj.x-rhon*r1q_dux;
				const float r1q_ey = r1q_inj.y-rhon*r1q_duy;
				const float r1q_ez = r1q_inj.z-rhon*r1q_duz;
				if(sqrt(fma(r1q_ex, r1q_ex, fma(r1q_ey, r1q_ey, r1q_ez*r1q_ez)))>1.0E-3f*r1q_mb+1.0E-12f&&hits[428]<0xF0000000u) atomic_inc(&hits[428]);
				if(fabs(r1q_inj.w)>1.0E-3f*r1q_mb+1.0E-12f&&hits[429]<0xF0000000u) atomic_inc(&hits[429]);
				const float r1q_d1 = (fwx*t1x+fwy*t1y+fwz*t1z)-r1q_fw1+r1q_m1;
				const float r1q_d2 = (fwx*t2x+fwy*t2y+fwz*t2z)-r1q_fw2+r1q_m2;
				if(fabs(r1q_d1)+fabs(r1q_d2)>1.0E-2f*r1q_mb+4.8E-7f*r1q_fwb&&hits[430]<0xF0000000u) atomic_inc(&hits[430]);
			}
			}
)+"#endif"+R( // FAC_R1Q_AN
		}
	}
)+"#endif"+R( // FAC_R1Q
)+"#ifdef FAC_REK_R3"+R(
	// ★★ BUCHUNG DER REKONSTRUKTION (23.09.2026). Der Block traegt der Zelle +rho*du an Impuls ein;
	// der Akkumulator [1..3] traegt den Impuls, den die WAND dem Fluid NIMMT (Vorzeichenkonvention
	// belegt an fwx = -(phi1*t1x+phi2*t2x) zwei Zeilen darueber, phi ist der aufgepraegte Impuls).
	// Also -rho*du. Die Groesse ist EXAKT, keine Naeherung: Summe_i c_i Df_i = rho*du, weil das
	// dritte Gittermoment von D3Q19 identisch verschwindet und der quadratische Term der Delta-Form
	// zum ersten Moment nichts beitraegt -- fuer JEDES du und JEDES s.
	// GEMESSEN vor dem Bau (Serie a2_bilanz, 23.09.): der Kernel-Akkumulator reproduziert die
	// ROHE Bilanzluecke FK.rx - Soll auf 1,2 % (eps +1e-4) und 1,8 % (eps -1e-3).
	// ★ Das Wort ROH steht hier seit 24.09., weil zwei Pruefer genau darueber gestolpert sind:
	// das Log druckt daneben die BEREINIGTE Luecke (0,82 % / 1,75 % / 14,5 %), und wer die gegen
	// diese Zahlen haelt, meldet einen Widerspruch, der keiner ist. Der einzige Arm, der
	// verfehlt (-1e-4, 13 %), ist auch der einzige nicht stationaere: seine Antriebskraft driftet
	// ueber das Fenster um 8,5 %, waehrend -1e-3 bei 0,1 % steht -- die Bilanzvoraussetzung gilt
	// dort nicht. Die Buchung steht also auf einer GEMESSENEN Bilanz, nicht auf einer Annahme.
	// rho ist das GEKLEMMTE rhon, dasselbe, das die Gewichte multipliziert hat: an 85 % der
	// markierten Besuche steht es auf der Klemme, ein roh nachgerechnetes waere bis 43 % zu gross.
	// ★★ DOPPELTERM-KORREKTUR (24.09.2026). VOR der Quellbuchung, damit der Block von oben nach
	// unten liest: erst zuruecknehmen, was doppelt drin ist, dann die Quelle EINMAL buchen.
	// fw = -(phi1*t1 + phi2*t2), und unter R3 ist phi == P (s == 0, kein Solve). P wird aber aus
	// fhn gebildet, das an den Marken schon das Df traegt -- also steckt -DP1*t1 - DP2*t2 bereits
	// in fw, ZUSAETZLICH zur expliziten Buchung -rho*du. Richtig waere -P_alt - rho*du; der
	// Ueberschuss ist genau -DP1*t1 - DP2*t2, er wird hier addiert.
	// GEMESSEN 24.09. (Slots 373..377, 8-mm-Fahrzeug): nur 8,3 % der markierten Besuche haben
	// |G11roh| < 1e-6 -- der Term ist am Fahrzeug gross, am Kanal kipp26 war er es nicht.
	// DP1/DP2 kommen EXAKT aus der Momentenschleife (Df je Wandlink nachgebildet), NICHT aus der
	// fuehrenden Ordnung rho*eps*G11roh: die haette den t2-Kanal (rho*eps*G12roh, an schiefen
	// Linkmengen bis sqrt(G11roh*G22roh)) und den in eps LINEAREN -1,5(du.s)-Anteil unterschlagen.
	const float rek_fv = fwx*t1x+fwy*t1y+fwz*t1z;
	const float rek_fb = fabs(fwx)+fabs(fwy)+fabs(fwz);
	fwx += rek_dp1*t1x + rek_dp2*t2x;
	fwy += rek_dp1*t1y + rek_dp2*t2y;
	fwz += rek_dp1*t1z + rek_dp2*t2z;
	// [378] ANGEWANDT-GEGEN-BEABSICHTIGT. Misst fw VOR und NACH den drei Zeilen darueber und
	// vergleicht die Differenz in t1 gegen rek_dp1. Faengt drei stille Fehler, die kein Zaehler
	// auf rek_gate fangen kann: (a) die Korrektur landet hinter fac_tau_acc und ist toter Code,
	// (b) sie steht ausserhalb der FAC_REK_R3-Insel, (c) VORZEICHEN verkehrt (-= statt +=) --
	// dann ist die Differenz -rek_dp1, also um 2*|rek_dp1| daneben. Toleranz: 1 % relativ plus
	// 4,8e-7*|fw| fuer die Ausloeschung (fw_t1 ist rund 100- bis 1000-mal groesser als dp1).
	if(rek_gate&&t%def_zaehl_takt==0ul) {
		const float rek_fn = fwx*t1x+fwy*t1y+fwz*t1z;
		// ★ TOLERANZ, zweimal berichtigt -- die Geschichte steht hier, weil beide Fassungen falsch
		// waren und die zweite schlimmer als die erste.
		// Fassung 1 (Erstbau): 1e-2*|dp1| + 4,8e-7*|fw.t1|. Erzeugte FEHLALARME. URSACHE ist NICHT,
		// wie hier zuerst stand, der t2-Anteil ueber t2.t1 -- der ist gemessen 1,4e-8 und traegt
		// fuenf Dekaden zu wenig. Die Ursache ist der AUSLOESCHUNGSBODEN: er hing an der PROJEKTION
		// fw.t1, der Rundungsboden der Differenz zweier Skalarprodukte skaliert aber mit |fw|.
		// An Treppenzellen ist |phi2| >> |phi1|, dort ist fw t2-dominiert und fw.t1 klein -- der
		// Boden zu klein, Fehlalarm.
		// Fassung 2 (Pruefbefund H1): 1e-2*(|dp1|+|dp2|). Behob den Fehlalarm, riss dafuer ein
		// groesseres Loch: bei verkehrtem Vorzeichen ist das Residuum exakt 2*|dp1|, und die
		// Toleranz uebersteigt das, sobald |dp2|/|dp1| > 199. Der Zaehler haette dann genau das
		// uebersehen, wofuer er gebaut ist. GEMESSEN am kipp26 (24.09.): an 93,3 % der Marken ist
		// |dp2| > 0,1*|dp1| -- das Regime ist die Regel, nicht die Ausnahme.
		// Fassung 3 (hier): Boden auf den VEKTORBETRAG von fw, relativer Teil auf dp1, t2 mit
		// eigenem, kleinem Schlupf. Faengt den Vorzeichenfehler bis |dp2|/|dp1| ~ 1e4 und erzeugt
		// im t2-dominierten Fall keinen Fehlalarm.
		const float rek_rel = fma(1.0E-4f, fabs(rek_dp2), 1.0E-2f*fabs(rek_dp1));
		const float rek_bod = 4.8E-7f*rek_fb;
		const float rek_tol = rek_rel+rek_bod;
		if(fabs((rek_fn-rek_fv)-rek_dp1)>rek_tol&&hits[378]<0xF0000000u) atomic_inc(&hits[378]);
		// [386] BEGLEITZAEHLER zu [378], Klasse von [334]: wie oft dominiert der Ausloeschungsboden
		// den relativen Teil? Wo er das tut, prueft [378] nichts, und seine Null waere ein
		// Scheinbeleg. GEMESSEN am kipp26 (24.09., Fassung 2 der Toleranz): 163 von 53 195 217 --
		// der Zaehler KANN feuern, er ist nicht konstruktiv still, aber der Fall ist selten.
		if(rek_bod>rek_rel&&hits[386]<0xF0000000u) atomic_inc(&hits[386]);
		if(fabs(rek_dp2)>0.1f*fabs(rek_dp1)&&hits[379]<0xF0000000u) atomic_inc(&hits[379]);
		// [387] BLINDHEITSSCHRANKE von [378], neu 24.09. Die Toleranz oben faengt einen
		// Vorzeichenfehler bis |dp2|/|dp1| ~ 1e4 und toten Code bis ~1e3 (eigener float-Versuch,
		// 200 000 Saetze je Stufe). Darueber ist [378] blind. [379] zaehlt ab Verhaeltnis 0,1 und
		// sagt deshalb NICHTS ueber diese Schranke. Dieser Zaehler tut es: steht er auf 0, ist die
		// Null von [378] ueber den ganzen gefahrenen Wertebereich belastbar.
		if(fabs(rek_dp2)>1.0E3f*fabs(rek_dp1)&&hits[387]<0xF0000000u) atomic_inc(&hits[387]);
		const float rek_kb = sqrt(fma(rek_dp1, rek_dp1, rek_dp2*rek_dp2));
		const float rek_bz = rek_rho*sqrt(fma(rek_dux, rek_dux, fma(rek_duy, rek_duy, rek_duz*rek_duz)));
		const float rek_q = rek_kb/fmax(rek_bz, 1.0E-30f);
		// ★ Pruefbefund M4 (24.09.): das oberste Fach war bei 0,1 OFFEN, die Groesse erreicht aber
		// bis 2,0 -- es konnte "ein Zehntel der Quelle" nicht von "doppelt so gross wie die Quelle,
		// Reibungspfad gedreht" unterscheiden. Dieselbe Lehre steht schon am Histogramm weiter oben.
		// Jetzt vier Grenzen, fuenf Faecher. ABNAHME: Summe [381..385] == [370].
		const uint rek_bq = rek_q<0.01f ? 0u : (rek_q<0.1f ? 1u : (rek_q<0.5f ? 2u : (rek_q<1.0f ? 3u : 4u)));
		if(hits[381ul+(ulong)rek_bq]<0xF0000000u) atomic_inc(&hits[381ul+(ulong)rek_bq]);
		// [388..392] HISTOGRAMM |DP_n|/(rho*|du|). Beantwortet, ob der Normalanteil des
		// Wandlink-Flusses gross genug ist, um im Druckpfad aufzufallen. Nur eine MESSUNG --
		// ob er dort ueberhaupt ankommt und ob er ungebucht ist, ist eine ANDERE Frage und
		// ausdruecklich noch nicht beantwortet. Grenzen 0,01/0,05/0,15/0,3.
		const float rek_qn = fabs(rek_dpn)/fmax(rek_bz, 1.0E-30f);
		const uint rek_bn = rek_qn<0.01f ? 0u : (rek_qn<0.05f ? 1u : (rek_qn<0.15f ? 2u : (rek_qn<0.3f ? 3u : 4u)));
		if(hits[388ul+(ulong)rek_bn]<0xF0000000u) atomic_inc(&hits[388ul+(ulong)rek_bn]);
		// [398] HOCH-3 (Nachpruefung 24.09.): unter S2 ist die Amplitude JE FACETTE UND SCHRITT 0 --
		// beim ersten Schritt immer, und jedes Mal, wenn die Schranke im Schritt davor gegriffen hat.
		// Dann ist rek_bz = 0, und beide Histogramme legen den Besuch stumm ins unterste Fach. Ohne
		// diesen Zaehler liest sich das als "der Anteil ist vernachlaessigbar", wo gar nichts geprueft
		// wurde -- dieselbe Klasse wie [334] und [386].
		if(rek_bz<=0.0f&&hits[398]<0xF0000000u) atomic_inc(&hits[398]);
		// [399] HOCH-4: die beiden Histogramme [373..377] und [381..385] widersprachen sich im
		// S2-Lauf (G11roh >= 1e-2 an 92 %, |DP|/(rho|du|) >= 1e-2 an 21 %). Zur fuehrenden Ordnung
		// ist DP1 = rho*d*G11roh, und zwar GLIEDWEISE vorzeichengleich je Wandlink -- es gibt keine
		// Ausloeschung. Beides zusammen kann nicht stimmen. Dieser Zaehler entscheidet es direkt,
		// statt zwei Histogramme gegeneinanderzuhalten: er vergleicht rek_dp1 gegen rho*d*rek_g11.
		// Das ist wichtig, weil die H1-Korrektur (P1 - rek_dp1) genau darauf steht.
		const float rek_soll = rek_rho*rek_g11*(rek_dux*t1x+rek_duy*t1y+rek_duz*t1z);
		if(fabs(rek_dp1-rek_soll)>0.2f*fmax(fabs(rek_soll), 1.0E-30f)&&hits[399]<0xF0000000u) atomic_inc(&hits[399]);
	}
	fwx -= rek_rho*rek_dux;
	fwy -= rek_rho*rek_duy;
	fwz -= rek_rho*rek_duz;
	// Normalprobe: du steht per Konstruktion tangential, also muss die Normalkomponente 0 sein.
	// Der Block hat sie bisher NICHT geprueft -- Befund 2 des Plans forderte Normal-Neutralitaet,
	// die Delta-Form erfuellt sie konstruktiv, aber unbelegt. Soll 0.
	// ★ 23.09. Pruefbefund M2: die Schranke war ABSOLUT (1e-6*rho) fuer eine Groesse, die mit eps
	// skaliert -- bei eps = 1e-7 haette sie nichts mehr geprueft und ihre Null gelesen sich wie ein
	// Beleg. Jetzt relativ zu |du|, und dimensionsrein (Geschwindigkeit gegen Geschwindigkeit).
	// Der unmarkierte Fall (alles 0) bleibt stumm, weil 0 > 0 falsch ist.
	if(t%def_zaehl_takt==0ul) {
		const float rek_dl = sqrt(rek_dux*rek_dux+rek_duy*rek_duy+rek_duz*rek_duz);
		if(fabs(rek_dux*nx+rek_duy*ny+rek_duz*nz)>1.0E-3f*rek_dl&&hits[372]<0xF0000000u) atomic_inc(&hits[372]);
	}
)+"#endif"+R( // FAC_REK_R3
)+R(
	const uxx a = 6ul*(uxx)fid; // Akkumulator: [1..3] Ist-Wandkraft (ungeklemmt == twe*t1; unter PEMA MODELLKRAFT: P gefiltert, P-Fluktuation laeuft als BB durch -- Audit 1/3)
	fac_tau_acc[a] += tw; fac_tau_acc[a+1ul] += fwx; fac_tau_acc[a+2ul] += fwy; fac_tau_acc[a+3ul] += fwz;
)+"#ifdef FACETTEN_ALPHA"+R(
)+"#ifdef FACETTEN_MASSE_ALLE"+R(
	// Stufe 3 kompensiert mit beta3 ueber ALLE Links: Sum q + Sum p = 6(S1.u_s) + beta3*Sum w_i,
	// und Sum w_i = 1. Der Rest ist wie unter Stufe 1/2 nur noch float-Rauschen -- die Zeile misst
	// das AUCH hier ehrlich und nicht per Konstruktion null (sie setzt beide Summanden getrennt).
	// ★ 04.09.2026 KORRIGIERT (Diff-Pruefung M1): hier stand "6*(S1.u_s) + beta3". Weil beta3 exakt
	// derselbe Float-Ausdruck mit umgekehrtem Vorzeichen ist, war die Summe BITGENAU NULL fuer jeden
	// Eingabewert -- das Abnahmekriterium "Delta-m bleibt float-ulp" konnte nicht reissen. Gebucht wird
	// jetzt die TATSAECHLICH verteilte Masse (Sum w_i in float ist 1,0000002384, nicht 1), also der
	// echte Rest von rund 2,2e-7*beta3 je Besuch.
	fac_tau_acc[a+4ul] += 6.0f*(S1x*usx+S1y*usy+S1z*usz) + masse_ist;
)+"#else"+R(
	fac_tau_acc[a+4ul] += fma(alph, S0, 6.0f*(S1x*usx+S1y*usy+S1z*usz)); // Delta-m-REST unter alpha: Soll ~float-ulp (Leck-Formel bleibt im #else als A/B-Referenz)
)+"#endif"+R( // FACETTEN_MASSE_ALLE
)+"#else"+R(
	fac_tau_acc[a+4ul] += 6.0f*(S1x*usx+S1y*usy+S1z*usz); // Delta-m-Leck (Gl. 13, komponentenweise)
)+"#endif"+R( // FACETTEN_ALPHA
)+"#ifdef FACETTEN_ALPHA"+R(
)+"#ifndef FACETTEN_ALPHA2"+R(
	// Werkzeugfalle Variante 3: get_opencl_c_code() laesst nach #if nur EIN Token zu -- deshalb
	// verschachtelte ifdef/ifndef statt "#if defined(A) && !defined(B)".
)+"#ifdef FACETTEN_MASSE_ALLE"+R(
	fac_tau_acc[a+5ul] += Sn1*s1+Sn2*s2+Snn*sn; // ★ S8 (04.09. abends): unter MASSE_ALLE ist beta3 impulsfrei -- KEIN alpha-Normalimpuls
)+"#else"+R(
	fac_tau_acc[a+5ul] += Sn1*s1+Sn2*s2+Snn*sn + alph*(S1x*nx+S1y*ny+S1z*nz); // Stufe 1: alpha-Normalimpuls ehrlich mitzaehlen
)+"#endif"+R( // FACETTEN_MASSE_ALLE
)+"#else"+R(
	fac_tau_acc[a+5ul] += Sn1*s1+Sn2*s2+Snn*sn;           // ALPHA2: via downgedatete Sn/Snn inkl. alpha
)+"#endif"+R( // FACETTEN_ALPHA2-Weiche
)+"#else"+R(
	fac_tau_acc[a+5ul] += Sn1*s1+Sn2*s2+Snn*sn;           // 3x3: REST-Normalinjektion (Soll ~0 bei Vollrang; N1 |Summe|<=5 je Torus-Lauf)
)+"#endif"+R( // FACETTEN_ALPHA-Weiche
	fac_tau_cnt[fid] += 1u;
)+"#ifdef FACETTEN_KDIAG"+R(
	{ const uxx k8 = 16ul*(uxx)fid; fac_kd[k8]+=ut; fac_kd[k8+1ul]+=tw; fac_kd[k8+2ul]+=twe; fac_kd[k8+3ul]+=fabs(P1); fac_kd[k8+4ul]+=s1; fac_kd[k8+5ul]+=phi1; fac_kd[k8+6ul]+=(rueckfall?1.0f:0.0f); fac_kd[k8+7ul]+=1.0f; fac_kd[k8+8ul]+=ut_ab; fac_kd[k8+9ul]+=yw_ab;
	  // ★ 04.09.2026 [10]/[11]: tw und Besuchszahl NUR ueber ANGEWANDTE Besuche. Grund: fac_tau[6i] (und
	  // damit y+ in yplus_facetten.csv und die Reportzeile 'Facetten-y+') summiert tw AUCH an
	  // Rueckfallzellen -- dort ist tw ein 'haette'-Wert, der nie aufgepraegt wurde. Am 8-mm-Fahrzeug
	  // sind das 42,8 % der Besuche. Jede y+-gestuetzte Aussage erbt diese Kontamination; mit [10]/[11]
	  // laesst sie sich erstmals BEZIFFERN statt nur vermuten.
	  fac_kd[k8+10ul]+=(pass2_an?tw:0.0f); fac_kd[k8+11ul]+=(pass2_an?1.0f:0.0f);
	  // ★ 05.09.2026 [12..15] VORZEICHENBEHAFTETER DRUCKREST (Bauplan Teil 1, Pruefbefund H-3). Der Betrags-
	  // zaehler 118..122 sagt: an 99 % der angewandten Besuche am klemmfreien kipp26 ist |2(rho-1)(S1.t1)|
	  // >= Ziel. Ob dieser Term sich ueber die Wand WEGHEBT (Treppenperiode symmetrisch, nur rho-Gradienten
	  // tragen netto) oder SYSTEMATISCH steht, kann ein Betrag nicht sagen. Deshalb hier, je Facette und mit
	  // Vorzeichen, NUR ueber angewandte Besuche (pass2_an, wie [10]/[11]):
	  //   [12] A = 2(rho-1)(S1.t1)      der isotrope Druckanteil IN P1; in R1 = B - P1 steht damit -A (Pruefer 05.09., Vorzeichen) (S1 ROH -- das Downdate aendert
	  //                                  nur G/Sn/Snn; t1 = die Basis, gegen die geloest wurde)
	  //   [13] |A|                       Betragssumme -- A/Sum|A| nahe 1 = systematisch, nahe 0 = hebt sich weg
	  //   [14] B = -def_fac_tau*twe      das Wandschubziel, IM KERNEL gebildet (def_fac_tau ist auf 4 Stellen
	  //                                  gerundet emittiert -- eine Host-Rekonstruktion waere eine zweite Wahrheit)
	  //   [15] C = 2(S1.t1)              die reine Geometrie; A/C = mittleres (rho-1) je Klasse, unabhaengige Probe
	  // NULLBEWEIS (Host): achsparallele Facetten MIT 5er-Linkmenge haben S1 = (0,0,+-1/6) und t1.n = 0
	  // bitgenau -> [12],[13],[15] == 0.0f BITGENAU. Nur diese Konfiguration, nicht der 0,99-Eimer (53ceb50).
	  { const float b1_kd = S1x*t1x+S1y*t1y+S1z*t1z;
	    const float a_kd = pass2_an ? 2.0f*(rhon-1.0f)*b1_kd : 0.0f;
	    fac_kd[k8+12ul]+=a_kd; fac_kd[k8+13ul]+=fabs(a_kd);
	    fac_kd[k8+14ul]+=(pass2_an?-def_fac_tau*twe:0.0f); fac_kd[k8+15ul]+=(pass2_an?2.0f*b1_kd:0.0f); } } // ★ S1 (04.09. abends): pass2_an statt !rueckfall -- unter KRAFT=2 ist pass2_an ueberall false, rueckfall nicht // [8]/[9]: Abtastwerte (== ut/yw ohne FACETTEN_NACHBAR) // ★ Klassen-Diagnostik: je Facette akkumuliert, Host mittelt je Treppenklasse (Iron Rule 3, Weg-1-Plan Stufe 0)
)+"#endif"+R( // FACETTEN_KDIAG
	if(rueckfall&&t%def_zaehl_takt==0ul&&hits[69]<0xF0000000u) atomic_inc(&hits[69]); // Slot 69: Rueckfall-Buchung (P-only), saettigend; Host prueft 69 == 13+15+64(+10+16 unter SATGATE)
)+"#ifdef FACETTEN_RDIAG"+R(
	// ★★ 07.09.2026 RUECKFALL-DIAGNOSE (CFD_FAC_RDIAG, Planungsagent-Schritt K1).
	// ANLASS: fac_kd[12..15] und die Slots 118..122 sind auf pass2_an gegatet und damit an genau
	// den Facetten BLIND, die zu 100 % zurueckfallen -- der Einzellink-Klasse (21,5 % Kugel,
	// 33,3 % kipp26). Solange das so ist, weiss niemand, ob deren |phi1|/twe (3,31 an der Kugel,
	// 27,8 am kipp26) ein REIBUNGSdefekt ist oder fehlgebuchter WANDDRUCK -- am kipp26 meldet der
	// Report den Rest-Druckterm in 99,2 % der ANGEWANDTEN Besuche >= Ziel, und ueber die
	// zurueckgefallenen sagt er nichts.
	// [136..140] |2(rho-1)(S1.t1)| / |Ziel| an RUECKFALLbesuchen, Dekaden 0,001/0,01/0,1/1
	// [141]/[142] Vorzeichen desselben Terms (hebt er sich weg oder steht er systematisch?)
	// [143] Nenner: gesampelte Rueckfallbesuche mit Ziel (twe > 0)
	// [144..148] / [150..154] s1_soll/u_t = -(def_fac_tau*twe+P1)/(G11roh*u_t), Grenzen 0/0,1/0,5/1,0.
	//            GETRENNT ueber den KASKADENZWEIG (Auditor A, Befund 1). DIE ERSTE FASSUNG TRENNTE
	//            NACH G11 GEGEN G11roh UND WAR EIN STILLER NO-OP: FACETTEN_ALPHA2 wird nur emittiert,
	//            wenn MASSE_ALLE == 0 (lbm.cpp), und ohne dieses Downdate ist G11 bitidentisch mit
	//            G11roh. Im gemessenen Arm (MASSE_ALLE=3) war 'G11roh > 0 und G11 == 0' damit
	//            unerfuellbar, [150..154] konstruktiv leer, und ALLE Rueckfaelle landeten im Eimersatz
	//            'der Loeser hat gerechnet' -- darunter 43,1 % reine Einzellink-Faelle, fuer die er gar
	//            nichts gerechnet hat. zweig (2258) traegt die REALE Kaskade: [144..148] zweig > 0, also
	//            Gate-Rueckfall -- der Loeser hat dort GERECHNET und ein Gate hat verworfen, der
	//            Quotient ist der Wert, den er setzen wollte. [150..154] zweig == 0 (Rang-0-Ausstieg,
	//            link): dort ist der unter der Massen-Nebenbedingung erreichbare Unterraum LEER
	//            (kernel.cpp:2151-2163, Befund 25.08.), R1/G11roh beantwortet also nur die
	//            HYPOTHETISCHE Frage "was waere ohne diese Nebenbedingung noetig" -- kein Sollwert.
	//            Wer beide Saetze addiert, vermischt eine Messung mit einer Hypothese.
	// [149] Rueckfallbesuche mit G11roh ~ 0 (c_1 parallel n, kipp45-Klasse) -- dort gibt es keine
	//       tangentiale Injektionsmoeglichkeit und auch keine BB-Bremse; s1_soll ist undefiniert.
	// BITNEUTRAL: nur Zaehler, kein Zugriff auf fhn. Abnahme = Feld-Hash gegen einen AUS-Arm
	// aus DEMSELBEN Binary. (Die Begruendung dafuer ist NICHT, dass Kommentartext das Compilat
	// aendere -- das ist widerlegt, R() ist #__VA_ARGS__ und der Praeprozessor entfernt Kommentare
	// VOR der Stringifizierung, kernel.hpp:4. Sondern: das Einschalten fuegt echten Code hinzu,
	// und am 07.09. brach ein Hash zwischen zwei Binaries mit identischem OpenCL-Text -- Ursache
	// ungeklaert, Kandidat FMA-Kontraktion im Host bei -O. Deshalb: AUS-neu gegen AN-neu.)
	if(rueckfall&&t%def_zaehl_takt==0ul&&def_fac_tau*twe>0.0f) {
		const float b1_rf = S1x*t1x+S1y*t1y+S1z*t1z;
		const float zi_rf = fabs(def_fac_tau*twe);
		const float a_rf = 2.0f*(rhon-1.0f)*b1_rf;
		const float rr_rf = zi_rf>0.0f ? fabs(a_rf)/zi_rf : 0.0f;
		const uint b_rf = rr_rf<0.001f?0u:(rr_rf<0.01f?1u:(rr_rf<0.1f?2u:(rr_rf<1.0f?3u:4u)));
		if(hits[136u+b_rf]<0xF0000000u) atomic_inc(&hits[136u+b_rf]);
		if(a_rf>0.0f) { if(hits[141]<0xF0000000u) atomic_inc(&hits[141]); }
		if(a_rf<0.0f) { if(hits[142]<0xF0000000u) atomic_inc(&hits[142]); }
		if(hits[143]<0xF0000000u) atomic_inc(&hits[143]);
		if(G11roh>1e-8f) {
			const float s1s_rf = -(def_fac_tau*twe+P1)/G11roh;
			const float q_rf = ut>1e-12f ? s1s_rf/ut : 0.0f;
			const uint b_s = q_rf<0.0f?0u:(q_rf<0.1f?1u:(q_rf<0.5f?2u:(q_rf<1.0f?3u:4u)));
			if(zweig>0u) { if(hits[144u+b_s]<0xF0000000u) atomic_inc(&hits[144u+b_s]); }
			if(zweig==0u) { if(hits[150u+b_s]<0xF0000000u) atomic_inc(&hits[150u+b_s]); }
		}
		if(G11roh<=1e-8f) { if(hits[149]<0xF0000000u) atomic_inc(&hits[149]); }
	}
)+"#endif"+R(
)+"#ifdef FACETTEN_DIAGZ"+R(
	// ★ Iron Rule 3 (Heiko 2026-08-16): eingebaute Zwischenergebnis-Diagnostik. Die gewaehlte
	// Facette schreibt ihre komplette Kette jeden Schritt in einen 18er-Puffer; der Host sampelt
	// je Chunk in eine CSV -- Plausibilitaet an kleinen Faellen SELBST nachvollziehbar.
	if((float)fid==fac_diag[16]) { // float-Vergleich: Sentinel -1.0f matcht nie (IR3-Audit: (uint)(-1.0f) war UB und haette fid 0 stumm geloggt)
		fac_diag[0]=ut; fac_diag[1]=twe; fac_diag[2]=P1; fac_diag[3]=P2;
		fac_diag[4]=s1; fac_diag[5]=s2; fac_diag[6]=sn;
		fac_diag[7]=phi1; fac_diag[8]=phi2;
		fac_diag[9]=G11; fac_diag[10]=G22; fac_diag[11]=Snn;
		fac_diag[12]=Sn1; fac_diag[13]=Sn2;
		fac_diag[14]=(float)t; fac_diag[15]=rhon;
)+"#ifdef FACETTEN_ALPHA"+R(
		fac_diag[17]=alph;
)+"#endif"+R( // FACETTEN_ALPHA
)+"#ifdef FACETTEN_APG"+R(
		fac_diag[18]=fac_dpds;
)+"#endif"+R( // FACETTEN_APG
	}
)+"#endif"+R( // FACETTEN_DIAGZ
	return kraft; // ★ KRAFT (30.08.): Zellkraft an Rueckfallzellen, sonst (0,0,0)
} // apply_facette_imem()
)+"#endif"+R( // FACETTEN_IMEM

// ★ TODO 2 Schritt 4 -- rho speichern. NUR speichern.
//
// HIER STAND BIS 12.09.2026 ABENDS EINE QUANTISIERUNGSMESSUNG, UND SIE WAR EIN STILLER NO-OP.
// Sie las den eben geschriebenen Wert an DERSELBEN Adresse im SELBEN Kernel zurueck und binte
// |load_rho(store_rho(x)) - x| in Dekaden. Der Intel-Uebersetzer entfernt diesen Umlauf: er gibt
// den Registerwert zurueck, statt neu zu laden. Die Differenz ist damit konstruktiv null.
// BELEGT, nicht vermutet -- 8-mm-Fahrzeug r8_fp16b gegen das eigene Feld bei 500 ms:
//   gemeldet   100,00 % unter 1e-7, 0,00 % darueber
//   verlangt    40,14 % unter 1e-7, 58,98 % unter 1e-6, 0,88 % unter 1e-5
//              (aus |rho-1| ueber 56.081.419 Fluidzellen des Dumps, mal 2^-12)
// Auf der CPU lieferte DERSELBE Quelltext 28,24 / 56,87 / 14,85 % -- dort uebersetzt ein anderer
// Compiler und der Umlauf bleibt stehen. Genau deshalb ist die CPU-Sprosse allein kein Beweis.
// Der vermeintliche Nullbeweis in Slot 217 (historisch, 12.09.; seit 15.09. traegt 217 die Besuche von rho_rek_ebene) war ebenso wertlos: er zaehlte eine Null, die per
// Konstruktion null war. Ein Waechter, der nicht feuern KANN, ist kein Waechter.
//
// WAS STATTDESSEN GILT. Die Quantisierung wird dort gemessen, wo sie sichtbar ist, ohne dass ein
// Uebersetzer sie wegkuerzen kann: am FELD-DUMP auf dem Host (Iron Rule 5 -- an Felddaten, nicht
// an einem Registerwert). Am 4-mm-Nahfeld, 451.428.942 Fluidzellen: RMS 3,32e-7, max 5,63e-5.
// Im BINARY bleibt der Bereichswaechter an der TYPE_E-Lesestelle (Slots 210/211). Der liest einen
// Wert, den ein ANDERER Kernel-Launch geschrieben hat -- dieser Ladevorgang kann nicht entfallen.

)+"#ifdef KLEMM_BILANZ"+R(
)+R(uint klemm_dekade(const float a) { // ★ 15.09.2026 Klemmen S0b (KLEMMEN-STUFE0-PLAN.md §6): Eimer 0..5 fuer <1e-4, <1e-3, <1e-2, <1e-1, <1, >=1
	return a<1.0E-4f ? 0u : (a<1.0E-3f ? 1u : (a<1.0E-2f ? 2u : (a<1.0E-1f ? 3u : (a<1.0f ? 4u : 5u))));
}
)+R(void klemm_summe(global uint* hits, const uint slot, const float a) { // Festkomma-Summand q = round(a*S) mit a >= 0; wickelt ABSICHTLICH mod 2^32 (Host bildet Fensterdifferenzen)
	float b = a;
	if(b>16.0f||(as_uint(b)&0x7F800000u)==0x7F800000u) { b = 16.0f; if(hits[265]<0xF0000000u) atomic_inc(&hits[265]); } // Kappung, Soll 0 -- sonst sind die Summen Untergrenzen. ★ Audit 16.09.2026 (A-N6), nachgebessert nach Pruefbefund H4: EXPONENT-BITTEST statt negiertem Vergleich -- unter -cl-finite-math-only (opencl.hpp) darf der Uebersetzer "!(b<=16)" wieder zu "b>16" falten und NaN erneut durchlassen; derselbe Bittest steht im Lift. Fuer endliche Werte unveraendert. "b>16" ist fuer NaN falsch, und convert_uint_sat(NaN) liefert 0: der Summand waere still verschwunden, waehrend klemm_dekade(NaN) in Eimer 5 zaehlt.
	atomic_add((volatile global uint*)&hits[slot], convert_uint_sat(fma(b, def_klemm_s, 0.5f)));
}
)+R(float klemm_rho_roh(const float* f) { // rho VOR der Dichteklemme, Summenreihenfolge wie calculate_rho_u
	float rho = f[0];
	for(uint i=1u; i<def_velocity_set; i++) rho += f[i];
	return rho+1.0f;
}
)+R(bool klemm_randschale(const uxx n) { // Randschale der Dicke 2 als reiner Koordinatentest (keine Flag-Lesung): Nahfeld = Koppelrand + Auslass, Fernfeld = Domaenenrand
	const uint3 c = coordinates(n); // ★ 15.09.2026 S0b: die 18er-Nachbarsuche brachte Spill 1216/864 (B70/iGPU) in stream_collide zurueck -- Gate-Bisektion, siehe KLEMMEN-STUFE0-PLAN.md Nachtrag
	return c.x<2u||c.x+2u>=def_Nx||c.y<2u||c.y+2u>=def_Ny||c.z<2u||c.z+2u>=def_Nz;
}
)+"#endif"+R( // KLEMM_BILANZ
)+"#ifdef POSITIV"+R(
)+R(float pos_s(const float fh, const float fe, const float w, const float gk, const float cm) { // ★ 15.09.2026 Klemmen Stufe 1 P1b (KLEMMEN-STUFE1-PLAN.md §1.3): Beitrag der Population i zum Skalierungsfaktor s
	// fh, fe in Rechenform (f - w_i); gk = tau_i - w_i (Kandidatenschwelle); cm = m0 + 3 c_i.m (Momente des Nichtgleichgewichts g = f* - f_eq)
	const float g = (fh-fe)-w*cm; // G_i = g_i - w_i (m0 + 3 c_i.m): Nichtgleichgewicht ohne 0. und 1. Moment
	const float b = fh-g;         // B_i - w_i = f_eq,i + w_i cm; Positivitaet verlangt B_i + s G_i >= tau_i
	if(b<gk) return fe<-w ? -2.0f : -1.0f; // machtlos: schon die Basis liegt unter tau_i (-2: f_eq,i selbst negativ)
	if(fh<gk) return (b-gk)/(-g);          // Kandidat: s_i = (B_i - tau_i)/(-G_i), hier -G_i > 0
	return 2.0f;                           // kein Beitrag
}
)+R(void pos_summe(global uint* hits, const uint slot, const float a) { // ★ P1c: Festkomma-Summand wie klemm_summe, eigene Kappung Slot 294 (Stufe 0 behaelt 265)
	float b = a;
	if(b>16.0f||(as_uint(b)&0x7F800000u)==0x7F800000u) { b = 16.0f; if(hits[294]<0xF0000000u) atomic_inc(&hits[294]); } // ★ Audit 16.09.2026 (A-N6, H4): NaN/Inf ueber den Exponent-Bittest, siehe klemm_summe
	atomic_add((volatile global uint*)&hits[slot], convert_uint_sat(fma(b, def_klemm_s, 0.5f)));
}
)+"#endif"+R( // POSITIV
)+R(kernel void stream_collide)+"("+R(global fpxx* fi, global rhoxx* rho, global velxx* u, global uchar* flags, const ulong t, const float fx, const float fy, const float fz, const uint felder_voll, global uint* rho_clamp_hits // ) { // main LBM kernel
)+"#ifdef FORCE_FIELD"+R(
	, const global float* F, const global uint* f_maske // argument order is important (f_maske: F-Markerliste, 03.09.; im Vollfeld-Arm ungelesen)
)+"#endif"+R( // FORCE_FIELD
)+"#ifdef SURFACE"+R(
	, const global float* mass // argument order is important
)+"#endif"+R( // SURFACE
)+"#ifdef TEMPERATURE"+R(
	, global fpxx* gi, global float* T // argument order is important
)+"#endif"+R( // TEMPERATURE
)+"#ifdef FACETTEN"+R(
	, const global float* fac_geo, const global uint* fac_idx, global float* fac_tau_acc, global uint* fac_tau_cnt // C1b, Reihenfolge = add_parameters in allocate()
)+"#ifdef FACETTEN_EMA"+R(
	, global float* fac_us // EMA-Zustand (nur Arm 3/4 mit CFD_FAC_EMA)
)+"#endif"+R( // FACETTEN_EMA
)+"#ifdef FACETTEN_PEMA"+R(
	, global float* fac_pu // PEMA-Zustand (nur Arm 3/4 mit CFD_FAC_PEMA)
)+"#endif"+R( // FACETTEN_PEMA
)+"#ifdef FACETTEN_DIAGZ"+R(
	, global float* fac_diag // Diagnose-Facette (CFD_FAC_DIAGZ)
)+"#endif"+R( // FACETTEN_DIAGZ
)+"#ifdef FACETTEN_ELIBB"+R(
	, const global uchar* fac_q // ★ B1: q je Link (18 uchar je Facette; Position = Host-add-Reihenfolge)
)+"#endif"+R( // FACETTEN_ELIBB
)+"#ifdef FACETTEN_KDIAG"+R(
	, global float* fac_kd // ★ Klassen-Diagnostik (Position = nach fac_q, Host-add-Reihenfolge)
)+"#endif"+R( // FACETTEN_KDIAG
)+"#ifdef FACETTEN_NACHBAR"+R(
)+"#ifdef FAC_REK"+R(
	, global float* fac_nb // ★★ 23.09. Pruefbefund H1: ohne dies verwarf der Aufruf an apply_facette_imem den Qualifier, und der Bau haengt -w an -- keine Diagnose. ★ 23.09. spaet H-N2: in #ifdef gezogen, damit der AUS-Arm quelltextidentisch bleibt.
)+"#else"+R(
	, const global float* fac_nb
)+"#endif"+R(
)+"#endif"+R( // FACETTEN_NACHBAR
)+"#ifdef SGS_FDWAND"+R(
	, const global float* fac_wfd // ★ Geistermoden-Fix: w je Facettenzelle aus |S|_FD des Vorschritts (Position = nach fac_kd)
)+"#endif"+R( // SGS_FDWAND
)+"#ifdef SGS_BAND"+R(
	, const global uint* band_idx, global float* band_sbar // ★ 22.09. (Plan C, SGS_BAND_PI): im Pi-Modus ist das der EMA-ZUSTAND (6 float je Bandzelle, wird hier geschrieben), sonst read-only Sbar. ★ 08.09. SGS-BAND: Maske und Sbar (Betrag des zeitgemittelten Scherratentensors) der Wandlagen 2..N (Position = NACH fac_wfd, Host-add-Reihenfolge in alloc_sgs_band)
)+"#endif"+R( // SGS_BAND
)+"#endif"+R( // FACETTEN
)+R( TS_P
)+") {"+R( // stream_collide()
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute stream_collide() on halo
)+"#ifdef SPARSE_TILES"+R(
	// FORK: Zellen in toten Tiles duerfen fi NIE anfassen. cell_base() gibt fuer sie defensiv 0
	// zurueck -- ohne diesen Ausstieg landen ihre Schreibzugriffe also in Slot 0 und ueberschreiben
	// die Daten einer echten aktiven Tile. Genau daran ist der erste T=8-Lauf divergiert (Cd 18.4).
	if(is_dead_tile(n, tile_slot)) return;
)+"#endif"+R(
	const uchar flagsn = flags[n]; // cache flags[n] for multiple readings
	const uchar flagsn_bo=flagsn&TYPE_BO, flagsn_su=flagsn&TYPE_SU; // extract boundary and surface flags
	if(flagsn_bo==TYPE_S||flagsn_su==TYPE_G) return; // if cell is solid boundary or gas, just return

	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices

	float fhn[def_velocity_set]; // local DDFs
	load_f(n, fhn, fi, j, t TS_A); // perform streaming (part 2)
)+"#ifdef POSITIV"+R(
	// ★ 15.09.2026 Klemmen Stufe 1 P1b: Slot 285 = Zelle mit negativer GELADENER Population (Nicht-E, vor MB/Facette), gezaehlt an den
	// Zaehlschritten t%def_zaehl_takt == 2 an Stichprobenzellen n%def_pos_sub == 0; Slot 289 = dieselbe Probe genau bei t == def_zaehl_takt+3 (Nachladeprobe fuer Haken 1).
	// Hier nur das FLAG, gezaehlt wird im Positiv-Block hinter der Kollision: die Atomics direkt nach load_f brachten Spill 448 (iGPU,
	// Nahfeld-Arme; Gate-Bisektion 15.09. abends), das Flag allein Spill 0 in allen Armen auf beiden Geraeten.
	const bool pneg_ = flagsn_bo!=TYPE_E&&((t%def_zaehl_takt==2ul&&n%(uxx)def_pos_sub==0u)|(t==(ulong)def_zaehl_takt+3ul))&&((fhn[0]<-def_w0)|(fhn[1]<-def_ws)|(fhn[2]<-def_ws)|(fhn[3]<-def_ws)|(fhn[4]<-def_ws)|(fhn[5]<-def_ws)|(fhn[6]<-def_ws)|(fhn[7]<-def_we)|(fhn[8]<-def_we)|(fhn[9]<-def_we)|(fhn[10]<-def_we)|(fhn[11]<-def_we)|(fhn[12]<-def_we)|(fhn[13]<-def_we)|(fhn[14]<-def_we)|(fhn[15]<-def_we)|(fhn[16]<-def_we)|(fhn[17]<-def_we)|(fhn[18]<-def_we));
)+"#endif"+R( // POSITIV

)+"#ifdef MOVING_BOUNDARIES"+R(
	if(flagsn_bo==TYPE_MS) apply_moving_boundaries(fhn, j, u, flags); // apply Dirichlet velocity boundaries if necessary (reads velocities of only neighboring boundary cells, which do not change during simulation)
)+"#endif"+R( // MOVING_BOUNDARIES

)+"#ifdef WANDFUNKTION"+R(
	// ★ Audit-Nacharbeit 3: TYPE_MS AUSGESCHLOSSEN. Vorher lief die WFB auch auf Zellen neben
	// BEWEGTEN Waenden -- NACH apply_moving_boundaries, dessen Injektion der Tausch aufs falsche
	// Linkpaar verschoben haette, und tau_w gegen das ABSOLUTE u_t. Bis die
	// Relativgeschwindigkeits-Erweiterung kommt, ist die bewegte Wand hart ausgenommen.
	if(flagsn_bo!=TYPE_S&&flagsn_bo!=TYPE_E&&flagsn_bo!=TYPE_MS) apply_wall_function(fhn, j, flags, rho_clamp_hits, t, true);
)+"#endif"+R( // WANDFUNKTION
)+"#ifdef FACETTEN"+R(
	// ★ C1b Stufe 2: gleicher Platz, gleiches Zellgate wie die z-WFB (beide schliessen sich per
	// Konstruktor-Fehler aus). Facetten-Lookup und Paar-Gates uebernehmen die Ortsaufloesung.
)+"#ifndef FACETTEN_IMEM"+R(
	if(flagsn_bo!=TYPE_S&&flagsn_bo!=TYPE_E&&flagsn_bo!=TYPE_MS) apply_facette(n, fhn, j, flags, fac_geo, fac_idx, fac_tau_acc, fac_tau_cnt, rho_clamp_hits, t);
)+"#else"+R(
	float3 fac_kraft = (float3)(0.0f,0.0f,0.0f); // ★ KRAFT: Zellkraft aus dem Wandmodell, unten auf fxn/fyn/fzn (Guo)
	if(flagsn_bo!=TYPE_S&&flagsn_bo!=TYPE_E&&flagsn_bo!=TYPE_MS) fac_kraft = apply_facette_imem)+"("+R(n, fhn, j, flags, fac_geo, fac_idx, fac_tau_acc, fac_tau_cnt, rho_clamp_hits, t
)+"#ifdef FACETTEN_EMA"+R(
		, fac_us
)+"#endif"+R( // FACETTEN_EMA
)+"#ifdef FACETTEN_PEMA"+R(
		, fac_pu
)+"#endif"+R( // FACETTEN_PEMA
)+"#ifdef FACETTEN_DIAGZ"+R(
		, fac_diag
)+"#endif"+R( // FACETTEN_DIAGZ
)+"#ifdef FACETTEN_ELIBB"+R(
		, fac_q
)+"#endif"+R( // FACETTEN_ELIBB
)+"#ifdef FACETTEN_NACHBAR"+R(
		, fac_nb
)+"#endif"+R( // FACETTEN_NACHBAR
)+"#ifdef FACETTEN_KDIAG"+R(
		, fac_kd
)+"#endif"+R( // FACETTEN_KDIAG
	)+");"+R(
)+"#endif"+R( // FACETTEN_IMEM
)+"#endif"+R( // FACETTEN
	float rhon, uxn, uyn, uzn; // calculate local density and velocity for collision
)+"#ifdef KLEMM_BILANZ"+R(
	uint kl = 0u; // ★ 15.09.2026 Klemmen S0b: Bit 0 Dichteklemme unten, Bit 1 oben, Bit 2 u-Klemme -- gebucht wird hinter SPONGE, wenn w feststeht
)+"#endif"+R( // KLEMM_BILANZ
)+"#ifndef EQUILIBRIUM_BOUNDARIES"+R(
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
)+"#ifdef RHO_CLAMP"+R(
	// ★ ZAEHLEN, nicht nur klemmen. Heikos Einwand 2026-08-09: die Klemme ist eine Kruecke, wenn der
	// Code oder die Parametrierung falsch ist -- richtig. Ein Treffer heisst: rho hat den
	// physikalischen Bereich verlassen, die Rechnung war dort schon kaputt, und die Klemme hat nur
	// die Explosion u = j/rho verhindert. Bleibt der Zaehler ueber den ganzen Lauf NULL, ist die
	// Klemme das, was sie sein soll: ein nie ausloesender Waechter. Sonst ist der Lauf kein Ergebnis.
	// Ich hatte diesen Waechter in defines.hpp beschrieben und nicht gebaut -- der lautlose No-op,
	// den dieses Projekt jagt, in meiner eigenen Klemme.
	// ★ 2026-08-25: GEGATET. Vorher ungegatet -- bei 2e8 Zellen wickelt uint nach rund 21
	// Schritten, und JEDE grosse Klemmzahl aus den Logs davor ist damit NICHT quantitativ
	// (auch meine eigene Rechnung "0,00039 Prozent der Zellaktualisierungen" vom 23.08.).
	// Jetzt Stichprobe wie alle uebrigen Zaehler: 1/100 der Schritte. Die Zahl ist damit ein
	// SAMPLE, kein Ereigniszaehler -- Legende in lbm.hpp sagt es.
	// [BERICHTIGT 15.09.2026, Klemmen-Plan §1: ueberholt -- seit Pruefbefund A1 UNGEGATET und saettigend, siehe die Zeile darunter.]
	if(rhon<=RHO_CLAMP_MIN) { if(rho_clamp_hits[0]<0xF0000000u) atomic_inc(&rho_clamp_hits[0]); } // saettigend statt gegatet (Pruefbefund A1)
	else if(rhon>=RHO_CLAMP_MAX) { if(rho_clamp_hits[1]<0xF0000000u) atomic_inc(&rho_clamp_hits[1]); }
)+"#ifdef RHO_HUELLE"+R(
	{ if(rhon<=def_rho_kons_lo&&rho_clamp_hits[298]<0xF0000000u) atomic_inc(&rho_clamp_hits[298]); else if(rhon>=def_rho_kons_hi&&rho_clamp_hits[299]<0xF0000000u) atomic_inc(&rho_clamp_hits[299]); } // ★ Z2f: Konsistenzhuelle (0,5/1,5) nur GEZAEHLT, geklemmt wird an der numerischen Huelle
)+"#endif"+R( // RHO_HUELLE
)+"#ifdef KLEMM_BILANZ"+R(
	if(rhon<=RHO_CLAMP_MIN) kl |= 1u; else if(rhon>=RHO_CLAMP_MAX) kl |= 2u; // Bedingung woertlich wie Slot 0/1 -- das traegt die Abnahme Summe(221..225) = [0]+[1]
)+"#endif"+R( // KLEMM_BILANZ
)+"#endif"+R(
)+"#else"+R( // EQUILIBRIUM_BOUNDARIES
	if(flagsn_bo==TYPE_E) {
)+"#ifdef RHO_RAND"+R(
		{ const uxx rr_ = rr_idx(n); // ★ C2b: R1; Slot 216 zaehlt Zugriffe ausserhalb (ungegatet, Soll 0; Besuchsbeleg Slot 211)
		  if(rr_>=(uxx)def_RR_N&&rho_clamp_hits[216]<0xF0000000u) atomic_inc(&rho_clamp_hits[216]);
		  rhon = load_rho(rho, rr_); }
)+"#else"+R(
		rhon = load_rho(rho,        n); // apply preset velocity/density
)+"#endif"+R( // RHO_RAND
		// ★ TODO 2 Schritt 4 -- BEREICHSWAECHTER an der einzigen rho-Lesestelle, ueber die der
		// Speicherinhalt in die RECHNUNG zurueckfliesst.
		//
		// WAS ER LEISTET, ehrlich und zweimal berichtigt (Audit-Schleife 12.09. abends, Pruefer A
		// und C). Er stand zuerst als Fang fuer die Typverwechslung da -- ein Kernel, dessen
		// rho-Parameter noch "global float* rho" heisst und zwei halbe Dichten als einen float
		// liest. DAS KANN ER NICHT: ein so gelesener Wert wird beim Zurueckschreiben wieder durch
		// den Halbwort-Dekoder gezogen und landet IN-RANGE. Nachgerechnet: rho = 1.0f als float
		// abgelegt und als zwei half gelesen ergibt 1,000000 und 1,0000572; selbst ein
		// fehlgelesenes 1e10 ergibt 1,0000000 und 1,000997. Die Typklasse faengt der Typ-Zensus in
		// lbm.cpp, und nur der. Dazu kommt: alle 18 verbliebenen float-Signaturen sind const,
		// schreiben also ohnehin nichts zurueck.
		// Was BLEIBT und was er wirklich ist: ein PHYSIK-Huellenwaechter. Die Schreiber von rho an
		// TYPE_E-Zellen garantieren eine Huelle -- RHO_CLAMP [0,5; 1,5] fuer stream_collide und
		// update_fields, das Tor im Kopplungs-Lift (0,5; 2,0). Verlaesst der gelesene Wert
		// [0,4; 2,1], hat einer dieser Schreiber seine Zusage gebrochen. Das ist eine Aussage,
		// die feuern KANN, und unter RHO_FP16 ist auch die obere Haelfte erreichbar (das Format
		// traegt bis 2,99902).
		// Der Bit-Test auf Exponent 0xFF steht hier, weil unter -cl-finite-math-only (opencl.hpp)
		// ein Vergleich Inf und NaN nicht faengt -- derselbe Kernel verwirft die Vergleichsform
		// weiter unten ausdruecklich ("Gross-Audit M") und nimmt dort denselben Bit-Test.
		//   Slot 210 UNGEGATET und saettigend: ein Nullbeweis, der den Anlauf mitsieht. Er kostet
		//   im sauberen Fall nichts -- der Vergleich ist ein Registertest, die Atomik feuert nur
		//   im Fehlerfall. Soll ueber den ganzen Lauf: 0.
		//   Slot 211 ist der Gegenbeweis, dass 210 auf dem ausgefuehrten Pfad liegt (ein Waechter
		//   ohne feuernden Besuchszaehler ist in diesem Projekt ein harter Fehler): EIN Schritt,
		//   damit die Zahl die TYPE_E-Zellen dieses Schritts ist und kein Mittel ueber viele.
		//   Es ist ein BESUCHSZAEHLER, kein Ist=Soll -- verglichen wird er auf dem Host mit nichts.
		if(((as_uint(rhon)&0x7F800000u)==0x7F800000u||rhon<=def_w210_lo||rhon>=def_w210_hi)&&rho_clamp_hits[210]<0xF0000000u) atomic_inc(&rho_clamp_hits[210]); // Soll 0 -- Huelle seit Z2e/Z2f emittiert (Vorgabe 0,4/2,1; TOR_HUELLE: Bildhuelle +- 16/32768; RHO_HUELLE: (0; 3))
		if(t==(ulong)def_zaehl_takt+2ul&&rho_clamp_hits[211]<0xF0000000u) atomic_inc(&rho_clamp_hits[211]); // Besuche
)+"#ifdef KLEMM_BILANZ"+R(
		// ★ Audit-Nachpruefung 16.09.2026, Befund M3: CFD_TOR_HUELLE verengt ZWEI Dinge -- das Lift-Tor (gespiegelt in [301]/[302])
		// UND diese Waechterhuelle. Ohne Spiegel war die zweite Haelfte des Schalters weiter nur durch eine Null belegt.
		// Ein Schritt je Zaehltakt, gleiche Gatterung wie der Besuchszaehler [211]; kein atomic, alle Threads schreiben denselben Wert.
		if(t==(ulong)def_zaehl_takt+2ul) { rho_clamp_hits[304] = (uint)fma(def_w210_lo, def_klemm_s, 0.5f); rho_clamp_hits[305] = (uint)fma(def_w210_hi, def_klemm_s, 0.5f); }
)+"#endif"+R( // KLEMM_BILANZ
		uxn  = load_u(u, n);
		uyn  = load_u(u, def_N+(ulong)n);
		uzn  = load_u(u, 2ul*def_N+(ulong)n);
		// ★ TODO 2 Schritt 4 (12.09.2026) -- Huellenwaechter fuer u, Zwilling zu Slot 210/211 bei rho,
		// aber gegen eine ANDERE Fehlerklasse. Bei rho faengt Slot 210 den Typverwechsler; bei u kann
		// er das nicht (dafuer ist der Typ-Zensus da). Hier geht es um die SAETTIGUNG: vstore_half_rte
		// kippt oberhalb |u*2^15| = 65504, also ab |u| = 1,99902, still nach +-inf -- und die
		// Projektlehre dazu steht schon im Code (setup.cpp: "dd_lauf01 kippte bei 0,15 s NICHT in nan,
		// sondern in die FP16-Saettigung" -- dort steht FP16C, gebaut ist FP16S; die Saettigungsfalle
		// gilt fuer beide, berichtigt 21.09.2026). Die Geschwindigkeitsklemme unten haelt +-0,57735 ein, aber
		// sie deckt nicht JEDEN Schreiber.
		// ★ BERICHTIGT 12.09. (Pruefagent, MITTEL): hier stand, der ungeklemmte Schreiber sei
		// drive_boundary_cubic_lift. Das war beim Schreiben richtig und ist es seit Slot 214 nicht
		// mehr -- dort steht jetzt ein eigenes Betragstor, dieser Waechter kann jene Klasse also
		// konstruktiv nicht mehr sehen. Was er WEITERHIN abdeckt: die Hostsaat (u wird an 56 Stellen
		// gesaet und mit write_to_device hochgeladen, ohne je durch die Klemme zu laufen) und
		// voxelize_mesh. NICHT gedeckt ist insert_rho_u_flags -- BERICHTIGT 12.09. abends (Pruefer A):
		// dessen Schreibziele sind genau die Halozellen, und stream_collide steigt bei is_halo(n) aus,
		// bevor es hier ankommt. Der Waechter sieht sie also nie. Heute latent, weil beide Domaenen
		// mit D=1 fahren und die Transferkernel gar nicht laufen; bei D>1 waere es eine Luecke.
		// Gegen Slot 214 ist er Redundanz -- und die ist hier gewollt, weil 214 im Kernel sitzt.
		// Schwelle 1,0 statt 1,99902: das ist Faktor 1,73 ueber der Klemme und Faktor 2,0 unter der
		// Saettigung -- der Waechter feuert, BEVOR das Wort kippt, nicht danach. Gemessenes Maximum im
		// 4-mm-Nahfeld bei 501 ms: 0,4764. Soll ueber den ganzen Lauf: 0.
		// Der Bit-Test auf Exponent 0xFF steht daneben, weil unter -cl-finite-math-only ein Vergleich
		// gegen inf nichts faengt -- dieselbe Begruendung wie bei rho zwei Zeilen darueber.
		if((fabs(uxn)>=1.0f||fabs(uyn)>=1.0f||fabs(uzn)>=1.0f
		   ||(as_uint(uxn)&0x7F800000u)==0x7F800000u||(as_uint(uyn)&0x7F800000u)==0x7F800000u||(as_uint(uzn)&0x7F800000u)==0x7F800000u)
		   &&rho_clamp_hits[212]<0xF0000000u) atomic_inc(&rho_clamp_hits[212]); // Soll 0, UNGEGATET
		if(t==(ulong)def_zaehl_takt+2ul&&rho_clamp_hits[213]<0xF0000000u) atomic_inc(&rho_clamp_hits[213]); // Besuche, sonst beweist die Null in 212 nichts
	} else {
		calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
)+"#ifdef RHO_CLAMP"+R(
		// ★★ ZAEHLER-FIX 2026-08-15 (Vorpruefer-Befund): die Zaehlung stand NUR im
		// #ifndef-EQUILIBRIUM_BOUNDARIES-Pfad -- der ist in JEDEM Build dieses Forks wegkompiliert
		// (EQUILIBRIUM_BOUNDARIES ist immer definiert). Die Klemme klemmte lautlos, und jede
		// "0 Treffer"-Meldung stammte von einem Zaehler, der im Binary nie existierte. Genau der
		// No-Op, vor dem der eigene Kommentar im anderen Zweig warnt.
		if(rhon<=RHO_CLAMP_MIN) { if(rho_clamp_hits[0]<0xF0000000u) atomic_inc(&rho_clamp_hits[0]); } // saettigend, siehe oben
		else if(rhon>=RHO_CLAMP_MAX) { if(rho_clamp_hits[1]<0xF0000000u) atomic_inc(&rho_clamp_hits[1]); }
)+"#ifdef RHO_HUELLE"+R(
		{ if(rhon<=def_rho_kons_lo&&rho_clamp_hits[298]<0xF0000000u) atomic_inc(&rho_clamp_hits[298]); else if(rhon>=def_rho_kons_hi&&rho_clamp_hits[299]<0xF0000000u) atomic_inc(&rho_clamp_hits[299]); } // ★ Z2f: Konsistenzhuelle (0,5/1,5) nur GEZAEHLT, geklemmt wird an der numerischen Huelle
)+"#endif"+R( // RHO_HUELLE
)+"#ifdef KLEMM_BILANZ"+R(
		if(rhon<=RHO_CLAMP_MIN) kl |= 1u; else if(rhon>=RHO_CLAMP_MAX) kl |= 2u; // Bedingung woertlich wie Slot 0/1 -- das traegt die Abnahme Summe(221..225) = [0]+[1]
)+"#endif"+R( // KLEMM_BILANZ
)+"#endif"+R( // RHO_CLAMP
	}
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
	float fxn=fx, fyn=fy, fzn=fz; // force starts as constant volume force, can be modified before call of calculate_forcing_terms(...)
)+"#ifdef FACETTEN_KRAFT"+R(
	fxn += fac_kraft.x; fyn += fac_kraft.y; fzn += fac_kraft.z; // ★ KRAFT: Wandmodell-Residuum als Volumenkraft (Guo-Kette unten, inkl. SGS-Guo-Korrektur)
)+"#endif"+R( // FACETTEN_KRAFT
	float Fin[def_velocity_set]; // forcing terms

// ★ F-Null-Read-Gate (Perf-Audit Achse 1, Rang 3, 2026-08-26): F ist an Nicht-Solid-Zellen
// konstant +0.0f (einziger aktiver Schreiber ist update_force_field, und der schreibt nur an
// TYPE_S; PARTICLES ist im Praeprozessor ausgeschlossen, der F-Waechter in initialize() prueft
// die Praemisse hart). Default AN via Emission; CFD_F_NUR_SOLID=0 stellt den Upstream-Read her.
)+"#ifdef FORCE_FIELD"+R(
)+"#ifndef F_NUR_SOLID"+R(
	{ // separate block to avoid variable name conflicts
		const float3 Fn = load3_F(F, f_maske, n); // FORK: bbox-bewusst
		fxn += Fn.x; fyn += Fn.y; fzn += Fn.z;
	}
)+"#endif"+R( // F_NUR_SOLID
)+"#endif"+R( // FORCE_FIELD

)+"#ifdef SURFACE"+R(
	if(flagsn_su==TYPE_I) { // cell was interface, eventually initiate flag change
		bool TYPE_NO_F=true, TYPE_NO_G=true; // temporary flags for no fluid or gas neighbors
		for(uint i=1u; i<def_velocity_set; i++) {
			const uchar flagsji_su = flags[j[i]]&TYPE_SU; // extract SURFACE flags
			TYPE_NO_F = TYPE_NO_F&&flagsji_su!=TYPE_F;
			TYPE_NO_G = TYPE_NO_G&&flagsji_su!=TYPE_G;
		}
		const float massn = mass[n]; // load mass
		     if(massn>rhon || TYPE_NO_G) flags[n] = (flagsn&~TYPE_SU)|TYPE_IF; // set flag interface->fluid
		else if(massn<0.0f || TYPE_NO_F) flags[n] = (flagsn&~TYPE_SU)|TYPE_IG; // set flag interface->gas
	}
)+"#endif"+R( // SURFACE

)+"#ifdef TEMPERATURE"+R(
	{ // separate block to avoid variable name conflicts
		uxx j7[7]; // neighbors of D3Q7 subset
		neighbors_temperature(n, j7);
		float ghn[7]; // read from gA and stream to gh (D3Q7 subset, periodic boundary conditions)
		load_g(n, ghn, gi, j7, t); // perform streaming (part 2)
		float Tn;
		if(flagsn&TYPE_T) {
			Tn = T[n]; // apply preset temperature
		} else {
			Tn = 0.0f;
			for(uint i=0u; i<7u; i++) Tn += ghn[i]; // calculate temperature from g
			Tn += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up gi (perturbation method / DDF-shifting)
		}
		float geq[7]; // cache f_equilibrium[n]
		calculate_g_eq(Tn, uxn, uyn, uzn, geq); // calculate equilibrium DDFs
		if(flagsn&TYPE_T) {
			for(uint i=0u; i<7u; i++) ghn[i] = geq[i]; // just write geq to ghn (no collision)
		} else {
)+"#ifdef UPDATE_FIELDS"+R(
			T[n] = Tn; // update temperature field
)+"#endif"+R( // UPDATE_FIELDS
			for(uint i=0u; i<7u; i++) ghn[i] = fma(1.0f-def_w_T, ghn[i], def_w_T*geq[i]); // perform collision
		}
		store_g(n, ghn, gi, j7, t); // perform streaming (part 1)
		fxn -= fx*def_beta*(Tn-def_T_avg);
		fyn -= fy*def_beta*(Tn-def_T_avg);
		fzn -= fz*def_beta*(Tn-def_T_avg);
	}
)+"#endif"+R( // TEMPERATURE

	{ // separate block to avoid variable name conflicts
)+"#ifdef VOLUME_FORCE"+R( // apply force and collision operator, write to fi in video memory
		const float rho2 = 0.5f/rhon; // apply external volume force (Guo forcing, Krueger p.233f)
)+"#ifdef KLEMM_BILANZ"+R(
		{ const float uxg_ = fma(fxn, rho2, uxn), uyg_ = fma(fyn, rho2, uyn), uzg_ = fma(fzn, rho2, uzn); // Guo-verschobenes u VOR der Klemme (identischer Ausdruck, IGC fasst zusammen)
		{ // ★ 15.09.2026 Klemmen Z2b (KLEMMEN-STUFE2-PLAN.md §2.1): Huellen VOR der Klemme zaehlen. 295 Komponentenhuelle |u_a| >= c_s (Soll = [28]
		  // solange die Komponentenklemme gilt), 296 Betragshuelle |u|^2 >= c_s^2 (groesste isotrope Kugel mit f_eq >= 0), 297 = 296 ohne 295 (Diagonalluecke).
		  const bool k295_ = fabs(uxg_)>=def_c||fabs(uyg_)>=def_c||fabs(uzg_)>=def_c;
		  const bool k296_ = uxg_*uxg_+uyg_*uyg_+uzg_*uzg_>=def_u2max;
		  if(k295_&&rho_clamp_hits[295]<0xF0000000u) atomic_inc(&rho_clamp_hits[295]);
		  if(k296_) { if(rho_clamp_hits[296]<0xF0000000u) atomic_inc(&rho_clamp_hits[296]); if(!k295_&&rho_clamp_hits[297]<0xF0000000u) atomic_inc(&rho_clamp_hits[297]); }
		}
		}
)+"#endif"+R( // KLEMM_BILANZ
)+"#ifdef U_BETRAG"+R(
		{ // ★ 15.09.2026 Klemmen Z2d (KLEMMEN-STUFE2-PLAN.md §2.1): BETRAGSklemme |u|^2 <= c_s^2 -- die groesste isotrope Kugel mit f_eq >= 0,
		  // parameterfrei; die Richtung bleibt, Delta j = w rho_c (s-1) u_roh (Stufe-0-Buchung gilt unveraendert). Kugel in Wuerfel: U_FP16-Marge und Lift-Schranke bleiben.
		  const float uxb_ = fma(fxn, rho2, uxn), uyb_ = fma(fyn, rho2, uyn), uzb_ = fma(fzn, rho2, uzn);
		  const float u2b_ = uxb_*uxb_+uyb_*uyb_+uzb_*uzb_;
		  const bool kb_ = u2b_>=def_u2max;
		  const float skb_ = kb_ ? sqrt(def_u2max/u2b_) : 1.0f;
		  uxn = uxb_*skb_; uyn = uyb_*skb_; uzn = uzb_*skb_;
		  if(kb_&&rho_clamp_hits[28]<0xF0000000u) atomic_inc(&rho_clamp_hits[28]); // Slot 28 = u-Klemme griff (hier: Betragshuelle)
)+"#ifdef KLEMM_BILANZ"+R(
		  if(kb_) kl |= 4u; // Bedingung woertlich wie Slot 28
)+"#endif"+R( // KLEMM_BILANZ
		}
)+"#else"+R( // U_BETRAG
		uxn = clamp(fma(fxn, rho2, uxn), -def_c, def_c); // limit velocity (for stability purposes)
		uyn = clamp(fma(fyn, rho2, uyn), -def_c, def_c); // force term: F*dt/(2*rho)
		uzn = clamp(fma(fzn, rho2, uzn), -def_c, def_c);
		// ★ 2026-08-25: Wirkpfad-Zaehler Slot 28. Die Geschwindigkeitsklemme war die EINZIGE
		// unbeobachtete Klemme im Kernel -- greift sie, ist der Impuls NICHT mehr erhalten
		// (f_eq traegt rho*u_geklemmt statt j+F/2). [BERICHTIGT 15.09.: UNGEGATET und saettigend, hier stand "gegatet".]
		if((fabs(uxn)>=def_c||fabs(uyn)>=def_c||fabs(uzn)>=def_c)&&rho_clamp_hits[28]<0xF0000000u) atomic_inc(&rho_clamp_hits[28]); // saettigend statt gegatet
)+"#ifdef KLEMM_BILANZ"+R(
		if(fabs(uxn)>=def_c||fabs(uyn)>=def_c||fabs(uzn)>=def_c) kl |= 4u; // Bedingung woertlich wie Slot 28
)+"#endif"+R( // KLEMM_BILANZ
)+"#endif"+R( // U_BETRAG
		calculate_forcing_terms(uxn, uyn, uzn, fxn, fyn, fzn, Fin); // calculate volume force terms Fin from velocity field (Guo forcing, Krueger p.233f)
)+"#else"+R( // VOLUME_FORCE
)+"#ifdef KLEMM_BILANZ"+R(
		{ // ★ 15.09.2026 Klemmen Z2b (KLEMMEN-STUFE2-PLAN.md §2.1): Huellen VOR der Klemme zaehlen. 295 Komponentenhuelle |u_a| >= c_s (Soll = [28]
		  // solange die Komponentenklemme gilt), 296 Betragshuelle |u|^2 >= c_s^2 (groesste isotrope Kugel mit f_eq >= 0), 297 = 296 ohne 295 (Diagonalluecke).
		  const bool k295_ = fabs(uxn)>=def_c||fabs(uyn)>=def_c||fabs(uzn)>=def_c;
		  const bool k296_ = uxn*uxn+uyn*uyn+uzn*uzn>=def_u2max;
		  if(k295_&&rho_clamp_hits[295]<0xF0000000u) atomic_inc(&rho_clamp_hits[295]);
		  if(k296_) { if(rho_clamp_hits[296]<0xF0000000u) atomic_inc(&rho_clamp_hits[296]); if(!k295_&&rho_clamp_hits[297]<0xF0000000u) atomic_inc(&rho_clamp_hits[297]); }
		}
)+"#endif"+R( // KLEMM_BILANZ
)+"#ifdef U_BETRAG"+R(
		{ // ★ 15.09.2026 Klemmen Z2d (KLEMMEN-STUFE2-PLAN.md §2.1): BETRAGSklemme |u|^2 <= c_s^2 -- die groesste isotrope Kugel mit f_eq >= 0,
		  // parameterfrei; die Richtung bleibt, Delta j = w rho_c (s-1) u_roh (Stufe-0-Buchung gilt unveraendert). Kugel in Wuerfel: U_FP16-Marge und Lift-Schranke bleiben.
		  const float uxb_ = uxn, uyb_ = uyn, uzb_ = uzn;
		  const float u2b_ = uxb_*uxb_+uyb_*uyb_+uzb_*uzb_;
		  const bool kb_ = u2b_>=def_u2max;
		  const float skb_ = kb_ ? sqrt(def_u2max/u2b_) : 1.0f;
		  uxn = uxb_*skb_; uyn = uyb_*skb_; uzn = uzb_*skb_;
		  if(kb_&&rho_clamp_hits[28]<0xF0000000u) atomic_inc(&rho_clamp_hits[28]); // Slot 28 = u-Klemme griff (hier: Betragshuelle)
)+"#ifdef KLEMM_BILANZ"+R(
		  if(kb_) kl |= 4u; // Bedingung woertlich wie Slot 28
)+"#endif"+R( // KLEMM_BILANZ
		}
)+"#else"+R( // U_BETRAG
		uxn = clamp(uxn, -def_c, def_c); // limit velocity (for stability purposes)
		uyn = clamp(uyn, -def_c, def_c); // force term: F*dt/(2*rho)
		uzn = clamp(uzn, -def_c, def_c);
		if((fabs(uxn)>=def_c||fabs(uyn)>=def_c||fabs(uzn)>=def_c)&&rho_clamp_hits[28]<0xF0000000u) atomic_inc(&rho_clamp_hits[28]); // saettigend statt gegatet // Slot 28, siehe oben
)+"#ifdef KLEMM_BILANZ"+R(
		if(fabs(uxn)>=def_c||fabs(uyn)>=def_c||fabs(uzn)>=def_c) kl |= 4u; // Bedingung woertlich wie Slot 28
)+"#endif"+R( // KLEMM_BILANZ
)+"#endif"+R( // U_BETRAG
		for(uint i=0u; i<def_velocity_set; i++) Fin[i] = 0.0f;
)+"#endif"+R( // VOLUME_FORCE
	}

)+"#ifdef UPDATE_FIELDS"+R(
)+"#ifdef EQUILIBRIUM_BOUNDARIES"+R(
	if(flagsn_bo!=TYPE_E) // only update fields for non-TYPE_E cells
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
	{
		)+"#ifdef RHO_RAND"+R(
		// ★ 15.09.2026 RHO_RAND C2b: rho nur in R1 und nur dort, wo es je Schritt gelesen wird (po_interior, x >= Nx-2 --
		// dieselbe konstruktive Obermenge wie RHO_SPARSAM). felder_voll Bit 0 hat hier KEINE Wirkung: Ausgaben lesen
		// rho_ausgabe_ebene. Slots 204/205 zaehlen wie unter RHO_SPARSAM genau einen Schritt.
		{ const bool rho_schreiben = (uint)(n%(uxx)def_Nx)+2u>=(uint)def_Nx;
		  if(t==(ulong)def_zaehl_takt+2ul&&rho_clamp_hits[rho_schreiben?205u:204u]<0xF0000000u) atomic_inc(&rho_clamp_hits[rho_schreiben?205u:204u]);
		  if(rho_schreiben) store_rho(rho, rr_idx(n), rhon); }
		)+"#else"+R(
		)+"#ifdef RHO_SPARSAM"+R(
		// ★ TODO 2 Schritt 1 (12.09.2026): rho wird nur noch dort geschrieben, wo es im NAECHSTEN
		// Schritt gelesen wird, plus an den Schritten, nach denen der Host das ganze Feld liest.
		// LESER von rho je feinem Schritt sind NUR po_reduce_mean und apply_pressure_outlet, und
		// beide lesen po_interior. Der Druckauslass ist die x_max-Flaeche (face_mask=0x2), die
		// Innenzelle entsteht aus einer 26er-Nachbarsuche um die Flaechenzelle (lbm.cpp:2717-2724)
		// -- jeder Kandidat hat damit x >= Nx-2. Die Bedingung ist also KONSTRUKTIV eine Obermenge
		// und reine Arithmetik (n = x+(y+z*Ny)*Nx, also ist n%def_Nx die x-Koordinate).
		// rho_voll kommt vom Host und ist 1 an jedem Schritt, nach dem das Feld gelesen wird.
		// NICHT betroffen: TYPE_E (schreibt dieser Zweig ohnehin nicht) und die u-Schreibstelle --
		// u hat eine viel groessere Leserschaft (deriv_reg an den 6 Nachbarn JEDER TYPE_E-Zelle,
		// sgs_fdwand, fac_nachbar_ab, schale_extract).
		{ // Die rho-Maske ist die Auslassschicht; im FERNFELD kommt die Schreibmasken-Box dazu, weil
		  // extract_plane_macros dort jeden Grobschritt rho AUF DEN FUENF ENTNAHMEEBENEN liest.
		  // RHO_SMBOX wird genau dann gesetzt, wenn die Box nicht die F-BBox ist (Fernfeld).
		  bool rho_schreiben = (felder_voll&1u)!=0u||(uint)(n%(uxx)def_Nx)+2u>=(uint)def_Nx;
		  )+"#ifdef RHO_SMBOX"+R(
		  if(!rho_schreiben) { const uint3 rxyz = coordinates(n);
			rho_schreiben = rxyz.x<2u||rxyz.x+2u>=(uint)def_Nx||rxyz.y<2u||rxyz.y+2u>=(uint)def_Ny||rxyz.z<2u||rxyz.z+2u>=(uint)def_Nz
				||(rxyz.x+2u>=(uint)def_SMX0&&rxyz.x<(uint)def_SMX0+(uint)def_SMNX+2u&&rxyz.y+2u>=(uint)def_SMY0&&rxyz.y<(uint)def_SMY0+(uint)def_SMNY+2u&&rxyz.z+2u>=(uint)def_SMZ0&&rxyz.z<(uint)def_SMZ0+(uint)def_SMNZ+2u); }
		  )+"#endif"+R( // RHO_SMBOX
		  // Wirkpfad-Zaehler (Iron Rule: ein Schalter ohne feuernden Zaehler ist ein harter Fehler).
		  // Slot 204 = uebersprungen, 205 = geschrieben. GENAU EIN SCHRITT (t == def_zaehl_takt), nicht
		  // jeder zaehl_takt-te: bei 251 gezaehlten Schritten x 64,9 Mio Zellen lief der 32-Bit-Zaehler in
		  // die Saettigung (0xF0000000) und die Prozentzahl war ein Artefakt -- am 12.09. genau so passiert
		  // und AN DEN ZAHLEN erkannt (beide Zaehler standen dicht ueber der Schwelle). Mit einem Schritt
		  // gilt 204+205 = aktive Zellen, also ein Ist=Soll statt einer Schaetzung.
		  // ★ +2 BERICHTIGT 12.09.: bei t == def_zaehl_takt traf die Zaehlung im FERNFELD genau einen
		  // erzwungenen Vollschreib-Schritt -- der Zeitschritt des Fernfelds laeuft dem Grobschritt um
		  // eins voraus, und 100 fiel damit auf ein Vielfaches der Sample-Kadenz. Der Zaehler meldete
		  // 0 % Ersparnis und der No-Op-Waechter brach den Lauf ab: ein falscher Alarm aus einem falsch
		  // gewaehlten Messzeitpunkt. Mit +2 liegt die Zaehlung in BEIDEN Domaenen mitten in der Periode.
		  if(t==(ulong)def_zaehl_takt+2ul&&rho_clamp_hits[rho_schreiben?205u:204u]<0xF0000000u) atomic_inc(&rho_clamp_hits[rho_schreiben?205u:204u]);
		  if(rho_schreiben) store_rho(rho, n, rhon); } // update density field
		)+"#else"+R(
		store_rho(rho, n, rhon); // update density field
		)+"#endif"+R( // RHO_SPARSAM
		)+"#endif"+R( // RHO_RAND
		)+"#ifdef U_SPARSAM"+R(
		// ★ TODO 2 Schritt 3 (12.09.2026): u wird nur noch dort geschrieben, wo es VOR dem naechsten
		// Vollschreiben gelesen wird. Die Leser von u je feinem Schritt und ihre Reichweite:
		//   deriv_reg  -- u an den SECHS Achsnachbarn JEDER TYPE_E-Zelle (kernel.cpp, Zweig
		//                 flagsn_bo==TYPE_E). TYPE_E liegt auf den Domaenenflaechen, die Leserzellen
		//                 also im Abstand 1 davon -> Randschale der Dicke 2 deckt sie.
		//   sgs_fdwand -- u an den sechs Achsnachbarn jeder Facettenzelle
		//   fac_nachbar_ab -- u am Link-Nachbarn einer Facettenzelle
		//                 Beide liegen in der F-BBox; um 2 dilatiert ist das eine Obermenge.
		//   apply_pressure_outlet -- u an po_interior, x >= Nx-2 -> von der Randschale gedeckt.
		//   schale_extract (N2F) -- 4^3-Bloecke ueber ~23 % der Domaene, ABER erst am Ende des
		//                 Grobschritts. Deshalb ist Bit 1 an jedem ratio-ten Schritt gesetzt; auf dem
		//                 letzten Substep wird u ueberall geschrieben. Daran haengt die Deckelung
		//                 des Gewinns auf (ratio-1)/ratio.
		// Der Test ist reine Arithmetik (Koordinaten gegen JIT-Defines), kein Speicherzugriff.
		{ bool u_schreiben = (felder_voll&2u)!=0u;
		  if(!u_schreiben) {
			const uint3 uxyz = coordinates(n);
			const bool rand = uxyz.x<2u||uxyz.x+2u>=(uint)def_Nx||uxyz.y<2u||uxyz.y+2u>=(uint)def_Ny||uxyz.z<2u||uxyz.z+2u>=(uint)def_Nz;
			const bool bbox = uxyz.x+2u>=(uint)def_SMX0&&uxyz.x<(uint)def_SMX0+(uint)def_SMNX+2u&&uxyz.y+2u>=(uint)def_SMY0&&uxyz.y<(uint)def_SMY0+(uint)def_SMNY+2u&&uxyz.z+2u>=(uint)def_SMZ0&&uxyz.z<(uint)def_SMZ0+(uint)def_SMNZ+2u;
			u_schreiben = rand||bbox;
		  }
		  // Wirkpfad-Zaehler: Slot 206 = uebersprungen, 207 = geschrieben (EIN Schritt, wie bei rho).
		  if(t==(ulong)def_zaehl_takt+2ul&&rho_clamp_hits[u_schreiben?207u:206u]<0xF0000000u) atomic_inc(&rho_clamp_hits[u_schreiben?207u:206u]);
		  if(u_schreiben) store3_u(u, n, (float3)(uxn, uyn, uzn));
		}
		)+"#else"+R(
		store3_u(u, n, (float3)(uxn, uyn, uzn)); // update velocity field
		)+"#endif"+R( // U_SPARSAM
	}
)+"#endif"+R( // UPDATE_FIELDS

	float feq[def_velocity_set]; // equilibrium DDFs
	calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
	float w = def_w; // LBM relaxation rate w = dt/tau = dt/(nu/c^2+dt/2) = 1/(3*nu+1/2)

)+"#ifdef SUBGRID"+R(
)+"#ifdef SGS_FDWAND"+R(
	// ★★ Geistermoden-Fix (B66/B69): an Facettenzellen kommt w aus dem FD-Kernel des Vorschritts
	// (geistermodenfreies |S|_FD) statt aus dem Pi-Tensor, den das Wandmodell kontaminiert.
	// SGS_WANDFREI hat VORRANG (Extremtest); Slot 39 zaehlt die Anwendung (t%100 wie ueblich).
	uint fdw_fid = 0xFFFFFFFFu;
)+"#ifdef SGS_BAND"+R(
	uint band_bid = 0xFFFFFFFFu; // ★ 08.09. SGS-BAND: Listenindex der Wandlagen 2..N, disjunkt zur Facettenmenge
)+"#endif"+R( // SGS_BAND
	{ uxx fbi_; if(flagsn_bo!=TYPE_S&&flagsn_bo!=TYPE_E&&flagsn_bo!=TYPE_MS&&f_bbox(n,&fbi_)) { fdw_fid = fac_fid(fac_idx, fbi_);
)+"#ifdef SGS_BAND"+R(
		if(fdw_fid==0xFFFFFFFFu) band_bid = band_fid(band_idx, fbi_); // nur wenn KEINE Facettenzelle -- die Mengen sind auf dem Host disjunkt gebaut
)+"#endif"+R( // SGS_BAND
	} }
)+"#endif"+R( // SGS_FDWAND
)+"#ifdef SGS_WANDFREI"+R(
	// ★★ TEST B der Rauwand-Diagnose (Laufzeitschalter CFD_SGS_WANDFREI, 2026-08-15): kein nu_t in
	// Zellen mit solidem FLAECHENnachbarn -- entscheidet, ob die gemessene Rauwand (k_s ~ 1 Zelle,
	// c_f 2,3-3,3x zu hoch) vom SGS-Modell oder vom Bounce-Back kommt. Das Gate sitzt VOR dem
	// Block und ueberspringt ihn (statt w hinterher zurueckzusetzen): so entfallen auch die
	// fneq-Produkte, SPONGE liest danach das laminare w und rampt korrekt, TRT leitet wm aus dem
	// finalen w ab. Wanderkennung ueber das etablierte Idiom (flags[j[i]]&TYPE_BO)==TYPE_S auf den
	// ALLEN 18 Nachbarn (seit 2a64a18 -- Diagonal-Facettenzellen!) -- bewusst NICHT ueber TYPE_MS,
	// denn ruhende Waende markieren keine MS-Zellen. Kostet 18 uchar-Reads je Zelle (im
	// FACETTEN-Build cache-neutral, iMEM liest dieselben Flags); der
	// Kontrollarm zahlt nichts, weil ungesetzt gar nicht emittiert wird.
	bool sgs_wand = false;
	for(uint i=1u; i<def_velocity_set; i++) sgs_wand = sgs_wand||(flags[j[i]]&TYPE_BO)==TYPE_S; // ★ WM-Blick D (MITTEL): vorher nur j[1..6] -- Diagonal-Facettenzellen (Kanten/Kugel/Fahrzeug) behielten die nu_t-Rueckkopplung im WANDFREI-Arm; jetzt alle 18
	if(sgs_wand&&t%def_zaehl_takt==0ul) atomic_inc(&rho_clamp_hits[6]); // R2: Wirkpfad-Nachweis (Befund-2-Rest), Slot 6, gegatet wie der WFB-Zaehler
	if(!sgs_wand)
)+"#endif"+R( // SGS_WANDFREI
)+"#ifdef SGS_FDWAND"+R(
	if(fdw_fid!=0xFFFFFFFFu) {
		w = fac_wfd[fdw_fid];
		if(t%def_zaehl_takt==0ul&&rho_clamp_hits[76]<0xF0000000u) atomic_inc(&rho_clamp_hits[76]); // Slot 76 (B70; 39 war der oberste SGS_DIAG-Wandlagen-Bin): FDWAND angewandt
)+"#ifdef SGS_VANDRIEST"+R(
		// ★★ VAN-DRIEST-DAEMPFUNG auf der FACETTEN-Architektur (CFD_SGS_VANDRIEST, 08.09.2026).
		// D = 1 - exp(-y+/A+), A+ = 26 (van Driest 1956, Literatur -- kein Handwert); nu_t <- nu_t * D^2.
		// ENTSCHEIDEND UND DER GRUND, WARUM DAS HIER GEHT: y+ kommt aus dem WANDMODELL, nicht aus dem
		// lokalen Strain. V1 schaetzte y+ = kappa*y^2*|S|/nu und multiplizierte dann mit demselben |S|
		// (FluidX3D/src/kernel.cpp:3079) -- eine Selbstreferenz, die an der Abloesung mit |S| -> 0 auch
		// y+ -> 0 und damit nu_t -> 0 trieb: WANDFREI genau dort, wo es divergiert. Der V1-Auditsatz
		// lautet "y+ aus lokalem Strain nullt sich an der Abloesung selbst". Hier dagegen:
		//   tw  = fac_tau_acc[6 fid]/fac_tau_cnt[fid] -- LAUFMITTEL der Wandschubspannung aus der
		//         Spalding-Kette des iMEM-Wandmodells (Eingang u_t, mit NACHBAR aus der zweiten
		//         Fluidzelle). Ein Zeitmittel kann der Momentanstroemung nicht folgen: die
		//         dynamische Rueckkopplung existiert konstruktiv nicht.
		//   y_w = fac_geo[8 fid + 3]                  -- Wandabstand aus der Voxelgeometrie.
		//   DEKLARIERT (Pruefagent 08.09., Befund 3): fac_tau_acc summiert tw ueber ALLE Besuche, auch
		//   Rueckfallbesuche mit "haette"-Werten (8 mm: 42,8 % der Besuche) -- dieselbe Reihe wie
		//   yplus_facetten.csv. Das Kontaminationsmass steht im Bericht (KDIAG-Vergleich gegen die
		//   angewandten Besuche); ein Wechsel auf die angewandte Reihe braeuchte 8 B je Facette ausserhalb
		//   von KDIAG und waere eine eigene Variable.
		//   LETZT-STICHPROBE (Befund 2): Slots 160..167 sind ein Zeitintegral ueber alle Zaehlslots ab der
		//   Sperre, der Host kennt aber nur den Endzustand -- ein Vergleich der beiden ist konstruktiv
		//   unscharf. Deshalb zaehlt jeder Slot zusaetzlich in Bank (t/100)&1 der Slots 170..185 und nullt
		//   im selben Slot die andere Bank (atomic_min auf 0, racefrei: niemand schreibt sie in diesem
		//   Slot). Nach dem Lauf traegt Bank (L/100)&1 GENAU den letzten Slot L.
		//   nu_t haengt weiter an |S|_FD aus dem u-Feld: zwei UNABHAENGIGE Groessen, D ist nur ein
		//   Gewicht, nu_t bleibt linear in |S| (V1 verhielt sich wie |S|^3).
		// Host-Spiegel derselben Formel: yplus_facetten.csv in setup.cpp.
		// 1/nu = 2*def_fac_Y (def_fac_Y = 0.5f/nu, unbedingt unter FACETTEN emittiert) -- NICHT
		// def_fac_nu verwechseln, das haengt am widerlegten u_w-Arm.
		if(t>=def_sgs_vd_ab) { // WARMLAUFSPERRE (08.09., iGPU-Befund Kanal N=108): tw ist ein Laufmittel seit t=0 und im Anlauf zu klein -> y+ zu klein -> D^2 zu klein -> ZU STARKE Daempfung genau in der Phase, in der WANDFREI nach 175 Schritten divergierte. Vorher: w unangetastet, nichts gezaehlt (Muster def_sgs_sism_ab).
		{	const uint vd_cnt = fac_tau_cnt[fdw_fid];
			if(vd_cnt>0u) {
				const float vd_tw = fac_tau_acc[6ul*(ulong)fdw_fid]/(float)vd_cnt;
				const float vd_yp = sqrt(fmax(0.0f, vd_tw))*fac_geo[8ul*(ulong)fdw_fid+3ul]*(2.0f*def_fac_Y);
				const float vd_d  = 1.0f-exp(-vd_yp*(1.0f/def_sgs_vd_aplus));
				const float vd_d2 = vd_d*vd_d;
				if(t%def_zaehl_takt==0ul) {
					const uint vd_bin = min(7u, (uint)(vd_d2*8.0f));
					if(rho_clamp_hits[160u+vd_bin]<0xF0000000u) atomic_inc(&rho_clamp_hits[160u+vd_bin]);
					if(rho_clamp_hits[168]<0xF0000000u) atomic_inc(&rho_clamp_hits[168]);
					const uint vd_bank = (uint)((t/100ul)&1ul);
					atomic_inc(&rho_clamp_hits[170u+8u*vd_bank+vd_bin]);
					for(uint vb=0u; vb<8u; vb++) atomic_min(&rho_clamp_hits[170u+8u*(1u-vd_bank)+vb], 0u);
				}
)+"#ifdef SGS_VANDRIEST_ANWENDEN"+R(
				const float vd_tau0 = 1.0f/def_w;
				const float vd_nut  = (1.0f/w-vd_tau0)*(1.0f/3.0f);
				w = 1.0f/(vd_tau0+3.0f*vd_d2*vd_nut);
)+"#endif"+R( // SGS_VANDRIEST_ANWENDEN
			} else if(t%def_zaehl_takt==0ul&&rho_clamp_hits[169]<0xF0000000u) atomic_inc(&rho_clamp_hits[169]);
		} }
)+"#endif"+R( // SGS_VANDRIEST
)+"#ifdef SGS_NUT_SKAL"+R(
		// ★★ DISKRIMINATOR-MESSARM (CFD_SGS_NUT_SKAL, 10.09.2026, Heiko-Go). KEIN Produktionsschalter.
		// DIE FRAGE: ist der gemessene SISM-Kraftgewinn Modellphysik -- oder nur die fehlende
		// Wanddaempfung? Beleg fuer die Frage: SISM senkt nu_t in Lage 1 um 85,2 % (Lagenmessung
		// 08.09., 4 mm), und die Feldpruefung 10.09. zeigt genau dort die unphysikalischen Zellen
		// (84,6 % der Ausreisser sind direkte Wandnachbarn, Grundrate 1,18 %).
		// DIESER ARM senkt nu_t am KLASSISCHEN Modell um denselben Faktor: gleiche Daempfung, KEINE
		// Scherungssubtraktion. Reproduziert er die Kraftaenderung, ist SISMs Gewinn keine Physik.
		// DER FAKTOR IST KEIN STELLKNOPF, SONDERN EIN MESSWERT -- er wird aus dem SISM-Lauf
		// ABGELESEN (1 - 0,852 = 0,148 fuer Lage 1). Deshalb Default 1,0 = aus = bitgleich, keine
		// Produktionsempfehlung und ein print_error gegen jede Kombination mit einem zweiten
		// nu_t-Senker (SISM, VANDRIEST=2) -- das waeren zwei Variablen in einem Lauf.
		// Idiom WOERTLICH wie der van-Driest-Zweig darueber: nu_t aus w, skalieren, w zurueck.
		// nu_mol als 0.5f/def_fac_Y (relativ genau auf 5e-8) statt (1/def_w-0.5)/3 -- letzteres
		// traegt bei tau0 = 0,50003 einen systematischen Bias von -0,14 % (4 mm) bis -0,85 %
		// (Fernfeld). Nachgerechnet in float32 vom Pruefagenten 10.09. Muster def_fac_nu.
		{	const float ns_tau0 = 1.0f/def_w, ns_numol = 0.5f/def_fac_Y;
			const float ns_nut = (1.0f/w-ns_tau0)*(1.0f/3.0f);
			const float ns_w = 1.0f/(ns_tau0+3.0f*def_sgs_nut_skal*ns_nut);
			if(t%def_zaehl_takt==0ul) {
				if(rho_clamp_hits[188]<0xF0000000u) atomic_inc(&rho_clamp_hits[188]); // Wirkpfad: Zweig besucht, MUSS gleich Slot 76 sein
				if(ns_nut>0.0f&&rho_clamp_hits[189]<0xF0000000u) atomic_inc(&rho_clamp_hits[189]); // es gab ueberhaupt ein nu_t zum Skalieren
				if(ns_w!=w&&rho_clamp_hits[190]<0xF0000000u) atomic_inc(&rho_clamp_hits[190]); // w hat sich WIRKLICH geaendert -- gegen den stillen No-Op
				const float ns_r = ns_nut/ns_numol; // nu_t in Vielfachen der molekularen Viskositaet
				// Saettigungsschranke AUCH hier, nicht nur an 188..190: der Wickelwaechter in
				// setup.cpp behandelt >= 0xF0000000 als gesaettigt und meldet, alles darunter aber
				// als print_error -- und print_error ist exit(1), also VOR der ganzen Endauswertung.
				// Ein ungeschuetztes Fach risse den Lauf ab, statt still zu saettigen.
				const uint ns_b = 191u+(ns_nut<=0.0f?0u:(ns_r<1.0f?1u:(ns_r<10.0f?2u:(ns_r<30.0f?3u:(ns_r<100.0f?4u:(ns_r<300.0f?5u:(ns_r<1000.0f?6u:7u)))))));
				if(rho_clamp_hits[ns_b]<0xF0000000u) atomic_inc(&rho_clamp_hits[ns_b]); // Summe 191..198 == Slot 188, solange nichts saettigt
			}
			w = ns_w;
		}
)+"#endif"+R( // SGS_NUT_SKAL
	} else
)+"#endif"+R( // SGS_FDWAND
	{ // Smagorinsky-Lilly subgrid turbulence model, source: https://arxiv.org/pdf/comp-gas/9401004.pdf, in the eq. below (26), it is "tau_0" not "nu_0", and "sqrt(2)/rho" (they call "rho" "n") is missing
		const float tau0 = 1.0f/w; // source 2: https://youtu.be/V8ydRrdCzl0
		float Hxx=0.0f, Hyy=0.0f, Hzz=0.0f, Hxy=0.0f, Hxz=0.0f, Hyz=0.0f; // non-equilibrium stress tensor
		for(uint i=1u; i<def_velocity_set; i++) {
			const float fneqi = fhn[i]-feq[i];
			const float cxi=c(i), cyi=c(def_velocity_set+i), czi=c(2u*def_velocity_set+i);
			Hxx += cxi*cxi*fneqi; //Hyx += cyi*cxi*fneqi; Hzx += czi*cxi*fneqi; // symmetric tensor
			Hxy += cxi*cyi*fneqi; Hyy += cyi*cyi*fneqi; //Hzy += czi*cyi*fneqi;
			Hxz += cxi*czi*fneqi; Hyz += cyi*czi*fneqi; Hzz += czi*czi*fneqi;
		}
		// ★★ 2026-08-25 GUO-KORREKTUR DES NICHTGLEICHGEWICHTSMOMENTS (Audit-Befund B).
		// Mit Guo-Antrieb ist das zweite Moment des Kraftterms (1-1/(2*tau))*(u_a F_b + F_a u_b);
		// das korrekte Nichtgleichgewichtsmoment lautet daher Pi = Sum cc(f-feq) + 0,5*(u_a F_b + F_a u_b)
		// (Krueger S.234, Guo 2002). Ohne diesen Term ist die SCHERRATE ueberall dort verzerrt, wo die
		// Volumenkraft nicht vernachlaessigbar ist -- und weil nu_0 hier praktisch 0 ist, verzerrt das
		// die GESAMTE Viskositaet. Im KANAL ist die Volumenkraft der Antrieb, die Delta-Reihe vom
		// 24.08. stand also auf einer verzerrten Scherrate. uxn traegt an dieser Stelle bereits den
		// F/(2*rho)-Schub, ist also die PHYSIKALISCHE Geschwindigkeit -- genau die, die hier gehoert.
)+"#ifdef SGS_GUO"+R(
)+"#ifdef VOLUME_FORCE"+R(
		{
			const float Qroh = sq(Hxx)+sq(Hyy)+sq(Hzz)+2.0f*(sq(Hxy)+sq(Hxz)+sq(Hyz));
			Hxx += uxn*fxn; Hyy += uyn*fyn; Hzz += uzn*fzn; // 0,5*(u_a F_b + F_a u_b), Diagonale = u_a F_a
			Hxy += 0.5f*(uxn*fyn+fxn*uyn); Hxz += 0.5f*(uxn*fzn+fxn*uzn); Hyz += 0.5f*(uyn*fzn+fyn*uzn);
			// ★ Pruefbefund 3-A (2026-08-25, HOCH): ohne Stichprobe feuert das je ZELLE je 100. Schritt --
			// bei 2e8 Zellen wickelt uint nach rund 21 Abtastungen, also rund 2100 Zeitschritten. Genau
			// die Rechnung, mit der derselbe Commit die Gatung von Slot 0/1 begruendet hat. Deshalb
			// zusaetzlich der multiplikative Hash von SGS_DIAG (jede 64. Zelle); die Prozente bleiben
			// erwartungstreu, der Zaehler bleibt im Bereich.
			if(t%def_zaehl_takt==0ul&&((n*2654435761ul)&4227858432ul)==0ul) { // Wirkpfad UND Groesse: relative Aenderung von |Pi| in vier Klassen
				const float Qneu = sq(Hxx)+sq(Hyy)+sq(Hzz)+2.0f*(sq(Hxy)+sq(Hxz)+sq(Hyz));
				const float qr = sqrt(Qroh), qn = sqrt(Qneu);
				const float rel = qr>0.0f ? fabs(qn-qr)/qr : (qn>0.0f ? 1.0f : 0.0f);
				atomic_inc(&rho_clamp_hits[60u+(rel<0.001f?0u:(rel<0.01f?1u:(rel<0.1f?2u:3u)))]);
			}
		}
)+"#endif"+R( // VOLUME_FORCE
)+"#endif"+R( // SGS_GUO
		const float Q = sq(Hxx)+sq(Hyy)+sq(Hzz)+2.0f*(sq(Hxy)+sq(Hxz)+sq(Hyz)); // Q = H*H, turbulent eddy viscosity nut = (C*Delta)^2*|S|, intensity of local strain rate tensor |S|=sqrt(2*S*S)
		w = 2.0f/(tau0+sqrt(sq(tau0)+0.76421222f*sqrt(Q)/rhon)); // 0.76421222 = 18*sqrt(2)*(C*Delta)^2, C = 1/pi*(2/(3*CK))^(3/4) = Smagorinsky-Lilly constant, CK = 3/2 = Kolmogorov constant, Delta = 1 = lattice constant
)+"#ifdef SGS_BAND"+R(
		// ★★ SGS-BAND, UMBAU 08.09.2026 nach dem gescheiterten Kipptest. Die erste Fassung ERSETZTE w
		// auf den Bandzellen durch das FD-w (wie an Facettenzellen) -- der 8-mm-Stressarm kippte damit
		// bei Schritt 392, und zwar auch OHNE SISM (Arm vb_d8_bandnosism, exakt dieselbe Zeit). Der
		// Grund ist physikalisch: an der FACETTENZELLE ist der Pi-Tensor vom Wandmodell kontaminiert
		// (Pi/FD = 2,3-3,4, gemessen 02.09.), dort ist der FD-Stencil eine REPARATUR. In Lage 2/3 ist
		// Pi sauber -- dort waere der FD-Stencil nur eine andere, glattere Diskretisierung, und die
		// traegt nicht.
		// JETZT: das Smagorinsky-w bleibt stehen, und der SISM-Abzug wird darauf angewandt. Weil
		// nu_t = c2*|S| linear in |S| ist, laesst sich der Abzug ohne |S| ausdruecken:
		//   nu_t,SISM = c2*max(0, |S| - Sbar) = max(0, nu_t - c2*Sbar).
		// Der Bandkernel liefert also nur noch Sbar je Bandzelle (band_sbar), nicht mehr ein fertiges w.
		if(band_bid!=0xFFFFFFFFu) {
			const float nut_b = (1.0f/w-tau0)*(1.0f/3.0f);            // nu_t aus dem eben gerechneten Smagorinsky-w
)+"#ifdef SGS_BAND_PI"+R(
			// ★★ 22.09.2026 PLAN C -- Pi-KONSISTENTE EMA (Tagesprotokoll B30/B32/B38). Gemessen am 8-mm-Fahrzeug (d5_band2_gdiag_8):
			// |S|_Pi/|S|_FD = 1,53 in Lage 2 (87 % der Zellen zwischen 1,2 und 3). Der FD-Modus zieht c2*Sbar_FD von c2*|S|_Pi ab und
			// entfernt damit nur ~60 % der mittleren Scherung (Klemme 6 % statt ~40 %). Hier: derselbe Schaetzer auf beiden Seiten.
			// S_ij = -3 Pi_ij / (2 rho tau_eff) mit tau_eff = 1/w des eben gerechneten Smagorinsky-w -- damit ist
			// c2*|S_Pi| EXAKT nut_b (Herleitung: tau_eff (tau_eff - tau0) = 0,76421222 sqrt(Q)/(4 rho); (tau_eff-tau0)/3 = 0,0636843 sqrt(Q)/(rho tau_eff)
			// = 0,030021 * 3 sqrt(2 Q)/(2 rho tau_eff)). EMA ueber T Schritte (alpha = 1/def_sgs_sism_T), Start 0 (kein Warmstart, wie Lage 1),
			// Phase 1 (t < def_sgs_sism_ab): kein Abzug, EMA laeuft mit. Reihenfolge: ALTES sb lesen -> nu_t bilden -> EMA schreiben.
			// Racefrei: band_bid ist je Zelle eindeutig (Maske+Praefixsumme), nur dieses Work-Item liest/schreibt band_sbar[6 bid..].
			// Kein u-Lesen an Lage 2..N mehr (der FD-Bandkernel entfaellt) -- der U_SPARSAM-Ausschluss gilt im Pi-Modus nicht.
			const float kS_ = -3.0f*w/(2.0f*rhon);
			const float Sp0=kS_*Hxx, Sp1=kS_*Hyy, Sp2=kS_*Hzz, Sp3=kS_*Hxy, Sp4=kS_*Hxz, Sp5=kS_*Hyz;
			const ulong k6b_ = 6ul*(ulong)band_bid;
			const float sb0=band_sbar[k6b_], sb1=band_sbar[k6b_+1ul], sb2=band_sbar[k6b_+2ul], sb3=band_sbar[k6b_+3ul], sb4=band_sbar[k6b_+4ul], sb5=band_sbar[k6b_+5ul];
			const float sbar_pi = sqrt(2.0f*(sq(sb0)+sq(sb1)+sq(sb2)+2.0f*(sq(sb3)+sq(sb4)+sq(sb5))));
			const float nut_n = t<def_sgs_sism_ab ? fmax(0.0f, nut_b) : fmax(0.0f, nut_b-0.030021f*sbar_pi); // Klemme ZWINGEND (wie Lage 1); Phase 1 wie der FD-Zweig geklemmt (Pruefbefund A-N1: nut_b kann an Q~0-Zellen 1 ulp negativ sein)
			const float a_pi = 1.0f/(float)def_sgs_sism_T;
			band_sbar[k6b_]     = fma(a_pi, Sp0-sb0, sb0);
			band_sbar[k6b_+1ul] = fma(a_pi, Sp1-sb1, sb1);
			band_sbar[k6b_+2ul] = fma(a_pi, Sp2-sb2, sb2);
			band_sbar[k6b_+3ul] = fma(a_pi, Sp3-sb3, sb3);
			band_sbar[k6b_+4ul] = fma(a_pi, Sp4-sb4, sb4);
			band_sbar[k6b_+5ul] = fma(a_pi, Sp5-sb5, sb5);
)+"#else"+R(
			const float nut_n = fmax(0.0f, nut_b-0.030021f*band_sbar[band_bid]); // Klemme wie in Lage 1 ZWINGEND (FD-Modus: Sbar_FD vom Bandkernel)
)+"#endif"+R(
			w = 1.0f/(tau0+3.0f*nut_n);
			if(t%def_zaehl_takt==0ul) {
				if(rho_clamp_hits[186]<0xF0000000u) atomic_inc(&rho_clamp_hits[186]); // Wirkpfad: Bandzelle behandelt
				if(nut_n<=0.0f&&rho_clamp_hits[187]<0xF0000000u) atomic_inc(&rho_clamp_hits[187]); // Klemme greift (Sbar >= |S|)
			}
		}
)+"#endif"+R( // SGS_BAND
	} // modity LBM relaxation rate by increasing effective viscosity in regions of high strain rate (add turbulent eddy viscosity), nu_eff = nu_0+nu_t
)+"#ifdef SGS_DIAG"+R(
	// ★ 03.09. (Pruefagent-Vorschlag 6): DIAG-Block HINTER das FDWAND-if/else gezogen. Er hing nur an w und tau0; im
	// else-Zweig lag er unter FDWAND fuer alle Facettenzellen brach (Wandlagen-Bins 35-48 leer, stiller No-Op).
	// Jetzt misst er das FINALE w -- an Facettenzellen also das FD-nu_t aus fac_wfd. tau0 = molekulares tau,
	// wortgleich zu 1/w VOR dem Smagorinsky-Zweig (def_w ist das molekulare w).
	{ const float tau0 = 1.0f/def_w;
		// ★ P0-DIAGNOSTIK (2026-08-23, GRENZSCHICHT-SGS-PLAN.md): nu_t/nu_0 als DEKADEN-
		// histogramm, Slots 28..32. Ein omega-Histogramm waere hier WERTLOS -- beide Gitter
		// stehen schon bei omega = 1,9999 (tau0 = 0,500028 nah, 0,500007 fern), die ganze
		// Verteilung laege in einem einzigen Bin. Aussagekraeftig ist nur das VERHAELTNIS,
		// und zwar logarithmisch: nu_t/nu_0 = (tau_neu - tau0)/(tau0 - 0,5). Diese eine Zahl
		// entscheidet, ob der Smagorinsky-Hebel ueberhaupt gebaut wird: traegt nu_t im
		// Fernfeld wirklich das Dreihundertfache der molekularen Viskositaet, lohnt der
		// Umbau -- sonst ist der Hebel tot, bevor er existiert.
		// GENAUIGKEIT: tau0 - 0,5 ist rund 2,8e-5 bei einer Zahl nahe 0,5, in float bleiben
		// davon gut zwei Stellen. Fuer Dekadengrenzen reicht das; auf den Einzelwert nicht bauen.
		// Physikfrei: w wird nicht angefasst, nur gezaehlt (t%100 gegen uint-Wickel).
		// PRUEFBEFUNDE 2026-08-23, eingearbeitet:
		//  WARMLAUFSPERRE (Pruefbefund 6, 2026-08-23): der Zaehler lief bisher AB SCHRITT 1, also
		//  auch waehrend der Einschwingphase -- am Fahrzeug stammten damit 40 Prozent der
		//  Stichproben aus einem Zustand, den der Lauf selbst als transient verwirft.
		//  def_sgs_diag_ab wird vom Host aus dem Warmlauf gesetzt.
		//  (3) t=0 ist ein Zaehlpunkt, dort ist fneq==0 und ALLES faellt in den <1-Bin -- das waren
		//      84 % des nahen und 63 % des fernen <1-Bins. Ausgeschlossen.
		//  (4) Der Filter (flags&TYPE_E)==0 wirft NICHT nur die kuenstlichen Randzellen hinaus:
		//      TYPE_MS (mitbewegte Wandnachbarn, 0x03) traegt das TYPE_E-Bit ebenfalls, also faellt
		//      die GESAMTE mitbewegte Fahrbahnlage mit heraus (262.337 fein / 90.702 grob bei 8 mm).
		//      Fuer die Fahrzeug-Grenzschichtfrage ist das richtig -- die Strasse gehoert nicht in
		//      die Statistik -- aber es ist eine ANDERE Wirkung als der Name nahelegt. Wer den
		//      Filter je lockert, holt die Fahrbahn unbemerkt herein (Pruefbefund 8, 2026-08-23).
		//  (9) uint wickelt. Deshalb nur jede 64. Zelle, und zwar ueber einen multiplikativen
		//      Hash statt ueber n&63: die Bitmaske entartet je nach Gittergroesse zu einer
		//      Ebenenschar (im Grobgitter war n&7 exakt (x+y) mod 8, in JEDER z-Lage identisch)
		//      und koennte damit ganze Wandorientierungen systematisch treffen oder verfehlen.
		//      Der Hash ist gittergroessen-unabhaengig (Pruefbefund 13, 2026-08-23).
		//  (11) Emission gegatet: ohne CFD_SGS_DIAG wird der Block gar nicht erst erzeugt.
		if(t>=def_sgs_diag_ab&&t>0ul&&t%def_zaehl_takt==0ul&&((n*2654435761ul)&4227858432ul)==0ul&&(flags[n]&TYPE_E)==0u) {
			const float nu0_ = tau0-0.5f;
			const float nut_ = 1.0f/w-tau0;
			const float rv_  = nut_/nu0_; // nu0_ > 0 ist Bauvoraussetzung, wird im Host geprueft
			const uint  b_   = rv_<1.0f ? 0u : (rv_<10.0f ? 1u : (rv_<100.0f ? 2u : (rv_<1000.0f ? 3u : 4u)));
			atomic_inc(&rho_clamp_hits[30u+b_]);
			/* ★ WANDNAHE LAGE, Slots 35..39 (2026-08-23). Der Dekadenhistogramm oben mittelt ueber
			   das GANZE Gitter -- Nachlauf und Freistrom eingeschlossen -- und beantwortet damit
			   nicht die Frage, um die es geht: traegt die anliegende Grenzschicht ZU VIEL oder ZU
			   WENIG modellierte Mischung? Der Massstab dafuer ist nicht die molekulare Viskositaet,
			   sondern das Gleichgewichtsprofil der Logschicht, nu_t/nu = kappa*y+ mit kappa = 0,41.
			   Bei y+ = 72 (Facettenmedian 8 mm) sind das rund 30; die Bingrenzen 5/15/30/60 klammern
			   diesen Wert ein. Deutlich darunter hiesse Modeled-Stress Depletion (zu wenig
			   wandnahe Mischung, verfruehte Abloesung) -- dann waere jede Senkung von C die
			   falsche Richtung. Deutlich darueber traegt die Ueberdissipations-Lesart.
			   EINSCHRAENKUNG, bewusst: "Lage 1" ist jede Zelle mit solidem 18er-Nachbarn, also auch
			   Unterboden, Raeder und abgeloeste Gebiete -- keine reine anliegende Grenzschicht.
			   Der Test kostet 18 uchar-Reads, aber nur auf den Zaehlschritten. */
			bool wand1_ = false;
			for(uint i=1u; i<def_velocity_set; i++) wand1_ = wand1_||(flags[j[i]]&TYPE_BO)==TYPE_S;
			if(wand1_) {
				const uint bw_ = rv_<5.0f ? 0u : (rv_<15.0f ? 1u : (rv_<30.0f ? 2u : (rv_<60.0f ? 3u : 4u)));
				atomic_inc(&rho_clamp_hits[35u+bw_]);
				/* Slots 40..44: dieselbe Lage, aber NUR mit vorwaertsgerichteter Stroemung.
				   In abgeloesten Gebieten GEHOERT viel nu_t hin -- sie mitzuzaehlen wuerde den
				   Ueberdissipations-Befund kuenstlich aufblasen. Dieser Teilsatz ist die
				   ehrliche Fassung: anliegende Stroemung, wo kappa*y+ ueberhaupt gilt. */
				if(uxn>0.0f) atomic_inc(&rho_clamp_hits[40u+bw_]);
				/* Slots 45..48 loesen den nach oben OFFENEN Bin auf. Ohne sie laesst sich nicht
				   sagen, ob eine Senkung von nu_t bei 20 oder bei 200 landet -- und genau davon
				   haengt ab, ob eine Aenderung der Konstante ueberhaupt in der richtigen
				   Groessenordnung waere (Pruefbefund 11). */
				if(rv_>=60.0f) atomic_inc(&rho_clamp_hits[45u+(rv_<120.0f?0u:(rv_<240.0f?1u:(rv_<480.0f?2u:3u)))]);
			}
		}
	}
)+"#endif"+R( // SGS_DIAG
)+"#endif"+R( // SUBGRID

)+"#ifdef SPONGE"+R(
	// ★★ DAEMPFUNGSZONE. Hebt nu in einem def_sponge_n Zellen breiten Streifen vor den TYPE_E-
	// Flaechen an (x-, x+, y-, y+, z+; der Boden z=0 ist Fahrbahn und bleibt aussen vor), mit
	// quadratischer Rampe bis Faktor def_sponge_a am Rand. Warum das der richtige Hebel ist und
	// drei Randumbauten es nicht waren, steht bei der Emission in lbm.cpp (Suchwort SPONGE).
	// Wirkt NACH SUBGRID auf dasselbe w, von dem TRT sein wm ableitet -- Lambda = 3/16 bleibt
	// also auch in der Zone erhalten, nur die Daempfung steigt.
	{
		const uint3 sxyz = coordinates(n);
		const uint sdx = min(sxyz.x, def_Nx-1u-sxyz.x);
		const uint sdy = min(sxyz.y, def_Ny-1u-sxyz.y);
		const uint sdz = def_Nz-1u-sxyz.z; // nur die Decke, nicht der Boden (rein GEOMETRISCH, kein TYPE_E-Test -- siehe lbm.cpp-Sponge-Kommentar)
		const uint sd = min(min(sdx, sdy), sdz);
		if(sd<def_sponge_n) {
			const float sr = 1.0f-(float)sd/(float)def_sponge_n; // 1 am Rand, 0 innen
			const float nu_l = fma(1.0f/w, 0.33333334f, -0.16666667f); // nu aus dem aktuellen w
			const float nu_s = nu_l*fma(sr*sr, def_sponge_a-1.0f, 1.0f);
			// ★★ KLEMME. Sie muss HIER stehen und nicht auf dem Host, weil nu_l oben aus dem bereits
			// von SUBGRID veraenderten w zurueckgerechnet wird: der Faktor trifft nu_0 + nu_t, und nu_t
			// kennt der Host nicht. (Den REIN LAMINAREN Anteil kennt er sehr wohl und meldet ihn beim
			// Start -- der Fall nu x1000 mal Zone x3000 aus dem 2026-08-09 waere schon dort aufgefallen.
			// Diese Klemme faengt den Anteil ab, der erst zur Laufzeit aus der Wirbelviskositaet kommt.)
			//
			// ★ KORRIGIERT nach Nachpruefung 2026-08-09: hier stand eine STABILITAETS-Begruendung
			// ("sqrt(2*nu_lat) ist die Diffusionsstrecke, darueber verlangt die Kollision einen
			// Transport, den das Streaming nicht liefert"). Das ist physikalisch falsch -- LBM ist fuer
			// 0 < w < 2 stabil, grosse tau sind ueberdaempft, aber stabil (im Grenzfall w->0 findet
			// einfach keine Kollision statt, also reine Advektion). nu_lat <= 0,5 ist KEINE
			// Stabilitaetsschranke.
			//
			// Die tragfaehige Begruendung ist GENAUIGKEIT: die in Chapman-Enskog vernachlaessigten
			// Terme gehen mit (tau-1/2)^2. tau=1 -> 0,25; tau=2 -> 2,25 (Faktor 9, vertretbar);
			// tau=21,8 (die gemessene Explosion) -> 453, also das 1800-fache von tau=1. Die Klemme ist
			// ein Genauigkeitsdeckel, kein Stabilitaetsschutz. w >= 0,5 heisst tau <= 2.
			// TRT bleibt konsistent: wp und wm werden weiter unten aus DIESEM w gebildet, Lambda = 3/16
			// gilt also auch in der Zone.
			// ★ 2026-08-25: Wirkpfad-Zaehler Slot 29. SPONGE war ein Mechanismus OHNE jeden
			// feuernden Zaehler -- nur eine laminare Vorschau beim Start. Gezaehlt wird der
			// Eintritt in die Zone; die Klemme selbst wird mitgezaehlt, indem der geklemmte
			// Fall den Zaehler ein zweites Mal erhoeht (Klemmanteil = 2*Klemme+Zone ueber Zone).
			const float w_roh_ = 1.0f/fma(3.0f, nu_s, 0.5f);
			w = fmax(w_roh_, def_sponge_wmin);
			// ★ Pruefbefund A3: vorher landeten Zone UND Klemme im SELBEN Slot (Z+K) -- daraus ist K
			// nicht rekonstruierbar, sp=2e6 ist zwischen (Z=2e6,K=0) und (Z=1e6,K=1e6) mehrdeutig.
			// Jetzt getrennte Slots, beide saettigend.
			if(rho_clamp_hits[29]<0xF0000000u) atomic_inc(&rho_clamp_hits[29]);
			if(w_roh_<def_sponge_wmin&&rho_clamp_hits[66]<0xF0000000u) atomic_inc(&rho_clamp_hits[66]);
		}
	}
)+"#endif"+R( // SPONGE
)+"#ifdef KLEMM_BILANZ"+R(
	if(kl!=0u) { // ★ 15.09.2026 Klemmen S0b (KLEMMEN-STUFE0-PLAN.md §2/§5/§6): hier steht w endgueltig fest (SGS und SPONGE vorbei, P-TRT liest w nur)
		// NUR SRT (lbm.cpp emittiert KLEMM_BILANZ nur unter SRT): unter TRT waere dj = wm*rho*du. Der tote #ifndef-EQUILIBRIUM_BOUNDARIES-Zweig setzt zwar kl-Bits, dort waeren load_u und Faktor 1 an TYPE_E aber falsch -- er ist in keinem Build aktiv (defines.hpp).
		uint kk = 4u; // Ortsklasse (ohne Flag-Lesung, spillfrei): K0 Facettenzelle > K1 TYPE_MS > K2 F-BBox ohne Facette ("fahrzeugnah") > K3 Randschale Dicke 2 > K4 Rest
		bool fb_in_ = false;
)+"#ifdef FORCE_FIELD"+R(
		{ uxx fbk_; fb_in_ = f_bbox(n, &fbk_);
)+"#ifdef FACETTEN"+R(
		  if(fb_in_&&flagsn_bo!=TYPE_S&&flagsn_bo!=TYPE_E&&flagsn_bo!=TYPE_MS) { if(fac_fid(fac_idx, fbk_)!=0xFFFFFFFFu) kk = 0u; }
)+"#endif"+R( // FACETTEN
		}
)+"#endif"+R( // FORCE_FIELD
		if(kk==4u) { if(flagsn_bo==TYPE_MS) kk = 1u; else if(fb_in_) kk = 2u; else if(klemm_randschale(n)) kk = 3u; }
)+"#ifdef RHO_CLAMP"+R(
		if((kl&3u)!=0u) { // Masse: dm = w*(rho_c - rho_roh); der Impuls bleibt, weil u = j/rho_c
			const float drho_ = rhon-klemm_rho_roh(fhn);
)+"#ifdef KLEMM_HAKEN3"+R(
			if(n%7u!=0u)
)+"#endif"+R( // KLEMM_HAKEN3
			{ if(rho_clamp_hits[221u+kk]<0xF0000000u) atomic_inc(&rho_clamp_hits[221u+kk]); }
			{ const uint dk_ = 236u+klemm_dekade(fabs(drho_)); if(rho_clamp_hits[dk_]<0xF0000000u) atomic_inc(&rho_clamp_hits[dk_]); }
			klemm_summe(rho_clamp_hits, ((kl&1u)!=0u ? 226u : 231u)+kk, fabs(w*drho_));
		}
)+"#endif"+R( // RHO_CLAMP
		if((kl&4u)!=0u) { // Impuls: dj = f*rho_c*(u_c - u_roh), f = w im Inneren, f = 1 an TYPE_E (REG_E = f_eq ohne Guo)
			float uxr_, uyr_, uzr_;
			if(flagsn_bo==TYPE_E) { uxr_ = load_u(u, n); uyr_ = load_u(u, def_N+(ulong)n); uzr_ = load_u(u, 2ul*def_N+(ulong)n); }
			else { float rr_; calculate_rho_u(fhn, &rr_, &uxr_, &uyr_, &uzr_); }
)+"#ifdef VOLUME_FORCE"+R(
			{ const float r2_ = 0.5f/rhon; uxr_ = fma(fxn, r2_, uxr_); uyr_ = fma(fyn, r2_, uyr_); uzr_ = fma(fzn, r2_, uzr_); }
)+"#endif"+R( // VOLUME_FORCE
			const float fak_ = (flagsn_bo==TYPE_E ? 1.0f : w)*rhon;
			const float djx_ = fak_*(uxn-uxr_), djz_ = fak_*(uzn-uzr_);
			if(rho_clamp_hits[242u+kk]<0xF0000000u) atomic_inc(&rho_clamp_hits[242u+kk]);
			{ const uint dk_ = 257u+klemm_dekade(fmax(fabs(uxn-uxr_), fmax(fabs(uyn-uyr_), fabs(uzn-uzr_)))); if(rho_clamp_hits[dk_]<0xF0000000u) atomic_inc(&rho_clamp_hits[dk_]); }
			if(djx_>0.0f) klemm_summe(rho_clamp_hits, 247u+kk, djx_); else if(djx_<0.0f) klemm_summe(rho_clamp_hits, 252u+kk, -djx_);
			if(djz_>0.0f) klemm_summe(rho_clamp_hits, 263u, djz_); else if(djz_<0.0f) klemm_summe(rho_clamp_hits, 264u, -djz_);
		}
	}
)+"#endif"+R( // KLEMM_BILANZ

)+"#if defined(EQUILIBRIUM_BOUNDARIES)&&defined(REGULARIZED_BOUNDARIES)"+R(
	// ★★ REGULARISIERTER GLEICHGEWICHTSRAND (Latt/Chopard 2006).
	// Statt f = f_eq wird an TYPE_E-Zellen f = f_eq + f_neq gesetzt, mit
	//    f_neq_i = -(3*rho*w_i/w) * [ c_i . S . c_i - tr(S)/3 ]
	// aus Pi_neq = -(2*rho*c_s^2/w)*S und f_neq_i = (w_i/(2 c_s^4)) * Q_i : Pi_neq, c_s^2 = 1/3.
	// Der reine Reset legt sonst alle 19 Verteilungen fest, wo hoechstens 5 zulaessig sind, und
	// verwirft jeden Schritt den gesamten Spannungstensor.
	//
	// S kommt aus Differenzen des FELDES u[] -- nicht aus den Verteilungen des Nachbarn, weil die
	// unter Esoteric Pull teilweise diesem selbst gehoeren und im selben Kernel-Start beschrieben
	// werden. u[] dagegen ist vom Vorschritt fertig und wird hier nur gelesen.
	//
	// ★ OHNE HILFSFELDER, und das ist keine Kosmetik: die erste Fassung legte zusaetzlich zu den
	// sechs bereits vorhandenen 19-Element-Feldern (fhn, feq, fhb, feb, Fin, j) noch du[9] und
	// fneq_reg[19] an. Der Intel-Uebersetzer blieb daraufhin beim Erzeugen des Kernels haengen --
	// reproduziert am 2026-08-08 mit 203 M UND mit 3 M Zellen, also am Kernel und nicht an der
	// Groesse. Hier deshalb sechs Skalare und vollstaendig ausgerollte Komponenten.
	float Sxx=0.0f, Syy=0.0f, Szz=0.0f, Sxy=0.0f, Sxz=0.0f, Syz=0.0f;
	if(flagsn_bo==TYPE_E) {
		// Nachbarn nur benutzen, wenn dort ECHTES Fluid steht (weder Solid noch Gleichgewichtsrand).
		// Das faengt zugleich den periodischen Umschlag ab: nach aussen zeigt der Nachbar auf die
		// gegenueberliegende Domaenenflaeche, und die ist selbst Rand.
		const uchar b1=flags[j[1]]&TYPE_BO, b2=flags[j[2]]&TYPE_BO, b3=flags[j[3]]&TYPE_BO;
		const uchar b4=flags[j[4]]&TYPE_BO, b5=flags[j[5]]&TYPE_BO, b6=flags[j[6]]&TYPE_BO;
		const bool px=b1!=TYPE_S&&b1!=TYPE_E, mx=b2!=TYPE_S&&b2!=TYPE_E; // +x, -x
		const bool py=b3!=TYPE_S&&b3!=TYPE_E, my=b4!=TYPE_S&&b4!=TYPE_E; // +y, -y
		const bool pz=b5!=TYPE_S&&b5!=TYPE_E, mz=b6!=TYPE_S&&b6!=TYPE_E; // +z, -z
		const float dxux=deriv_reg(u,      0ul, j[1], j[2], px, mx, uxn);
		const float dxuy=deriv_reg(u,    def_N, j[1], j[2], px, mx, uyn);
		const float dxuz=deriv_reg(u, 2ul*def_N, j[1], j[2], px, mx, uzn);
		const float dyux=deriv_reg(u,      0ul, j[3], j[4], py, my, uxn);
		const float dyuy=deriv_reg(u,    def_N, j[3], j[4], py, my, uyn);
		const float dyuz=deriv_reg(u, 2ul*def_N, j[3], j[4], py, my, uzn);
		const float dzux=deriv_reg(u,      0ul, j[5], j[6], pz, mz, uxn);
		const float dzuy=deriv_reg(u,    def_N, j[5], j[6], pz, mz, uyn);
		const float dzuz=deriv_reg(u, 2ul*def_N, j[5], j[6], pz, mz, uzn);
		Sxx=dxux; Syy=dyuy; Szz=dzuz;
		Sxy=0.5f*(dxuy+dyux); Sxz=0.5f*(dxuz+dzux); Syz=0.5f*(dyuz+dzuy);
	}
	const float trS3 = 0.33333334f*(Sxx+Syy+Szz);
	const float regf = -3.0f*rhon/w; // `w` ist hier die RELAXATIONSRATE und verdeckt die Gewichtsfunktion w(i)
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES && REGULARIZED_BOUNDARIES

)+"#ifdef PTRT"+R(
	// ★★ P-TRT, Purified TRT (Yu/Yu/Zhou/Chen/Yuan/Shu, arXiv:2602.06686), gebaut 10.09.2026.
	// Der GEISTANTEIL des symmetrischen Nichtgleichgewichts wird mit def_omega_g relaxiert statt
	// mit der Kollisionsrate w:  f^post = <Kollision> + (w - omega_g) * (Pg n),  n = f - feq.  Herleitung ohne Umweg:
	// SRT ist f - w*n, TRT ist f - wp*n+ - wm*n-; spaltet man den GERADEN Anteil in Hydro-
	// und Geistanteil und relaxiert den Geistanteil mit omega_g, bleibt genau dieser Term.
	// ★ DIESER FORK RECHNET SRT (defines.hpp:10), NICHT TRT (defines.hpp:19 auskommentiert).
	// Der Block stand zuerst im TRT-Zweig und war damit toter Code -- die Abnahme hat es am
	// 10.09. abends gefangen (Slot 199 = 0, Feld-Hash unveraendert). Er steht jetzt VOR der
	// Weiche, gilt also fuer beide Operatoren.
	//
	// DREI DINGE, DIE HIER SCHIEFGEHEN KOENNEN, und warum sie es nicht tun:
	//  (1) Die Momentensumme ist UNGEWICHTET (sum_j g_j n_j), die Rueckgabe GEWICHTET (w_i g_i ...).
	//      Beides gewichtet waere kein Projektor -- ein stiller Halbtreffer.
	//  (2) w_i ist die GEWICHTSFUNKTION, nicht die Relaxationsrate. Im TRT-Block verdeckt die
	//      lokale Rate `w` den Namen (siehe die Warnung an der regf-Zeile weiter oben), deshalb
	//      stehen hier die Makros def_w0/def_ws/def_we und NICHT w bzw. w(i).
	//  (3) KEINE Tabelle, KEINE Schleife, KEIN laufzeitindiziertes Feld. Der Kernel traegt schon
	//      sechs 19er-Felder; zwei weitere liessen den Intel-Uebersetzer beim Erzeugen haengen
	//      (Anmerkung am Regularisierungsblock). Ausgerollt kostet es nichts: ueber alle 19
	//      Richtungen gibt es nur SIEBEN verschiedene Korrekturwerte.
	//
	// Basis (korrigiert gegenueber A.7-A.9 des Preprints, dort fuer D3Q19 nicht orthogonal),
	// in der Linkreihenfolge dieses Forks, mit Gram diag(2, 4/3, 4/9):
	//   g1 = [ 1,-2,-2,-2,-2,-2,-2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	//   g2 = [ 0,-2,-2, 1, 1, 1, 1, 1, 1, 1, 1,-2,-2, 1, 1, 1, 1,-2,-2]
	//   g3 = [ 0, 0, 0,-1,-1, 1, 1, 1, 1,-1,-1, 0, 0, 1, 1,-1,-1, 0, 0]
	// Orthogonalitaet 4,2e-17, Projektor idempotent 1,1e-16, Masse/Impuls/Spannung/3. Momente
	// des Abzugs exakt null, ausgerollte Form gegen die Projektormatrix 1,8e-15 -- alles in
	// werkzeuge/vonneumann.py nachrechenbar.
	// ★ fmax, nicht die blanke Differenz (Kernel-Pruefer 10.09. nachts): der Abzug setzt den
	// Geist-Eigenwert auf 1-omega_g, UNABHAENGIG von w. Wo die lokale Rate unter omega_g faellt,
	// waere das eine VERSCHLECHTERUNG -- gemessene Schwelle nu_t/nu_0 > 452 im Nahfeld, > 1810 im
	// Fernfeld, und die aeusseren 15 der 64 Sponge-Zellen liegen darunter (w_Rand = 1,9185).
	// Mit fmax relaxiert der Geist mit min(w, omega_g) und damit nirgends langsamer als heute.
	// Kein Handwert, eine Zeile, ein FLOP.
	const float pt_k = fmax(0.0f, w-def_omega_g); // = 0 hiesse No-Op; Zaehler 201 deckt genau das auf
	const float pt_n0=fhn[0]-feq[0];
	const float pt_n1=fhn[1]-feq[1], pt_n2=fhn[2]-feq[2], pt_n3=fhn[3]-feq[3];
	const float pt_n4=fhn[4]-feq[4], pt_n5=fhn[5]-feq[5], pt_n6=fhn[6]-feq[6];
	const float pt_n7=fhn[7]-feq[7], pt_n8=fhn[8]-feq[8], pt_n9=fhn[9]-feq[9];
	const float pt_n10=fhn[10]-feq[10], pt_n11=fhn[11]-feq[11], pt_n12=fhn[12]-feq[12];
	const float pt_n13=fhn[13]-feq[13], pt_n14=fhn[14]-feq[14], pt_n15=fhn[15]-feq[15];
	const float pt_n16=fhn[16]-feq[16], pt_n17=fhn[17]-feq[17], pt_n18=fhn[18]-feq[18];
	const float pt_ger = pt_n1+pt_n2+pt_n3+pt_n4+pt_n5+pt_n6; // gerade Links (1..6)
	const float pt_kan = pt_n7+pt_n8+pt_n9+pt_n10+pt_n11+pt_n12+pt_n13+pt_n14+pt_n15+pt_n16+pt_n17+pt_n18;
	const float pt_m1 = pt_n0-2.0f*pt_ger+pt_kan;
	const float pt_m2 = -2.0f*(pt_n1+pt_n2)+(pt_n3+pt_n4+pt_n5+pt_n6)+(pt_n7+pt_n8)+(pt_n9+pt_n10)
	                  -2.0f*(pt_n11+pt_n12)+(pt_n13+pt_n14)+(pt_n15+pt_n16)-2.0f*(pt_n17+pt_n18);
	const float pt_m3 = -(pt_n3+pt_n4)+(pt_n5+pt_n6)+(pt_n7+pt_n8)-(pt_n9+pt_n10)+(pt_n13+pt_n14)-(pt_n15+pt_n16);
	const float pt_a1 = 0.5f*pt_k*pt_m1;   // durch die Gram-Diagonale 2
	const float pt_a2 = 0.75f*pt_k*pt_m2;  // durch 4/3
	const float pt_a3 = 2.25f*pt_k*pt_m3;  // durch 4/9
	const float pt_d0 = def_w0*pt_a1;
	const float pt_dx = def_ws*(-2.0f*pt_a1-2.0f*pt_a2);
	const float pt_dy = def_ws*(-2.0f*pt_a1+pt_a2-pt_a3);
	const float pt_dz = def_ws*(-2.0f*pt_a1+pt_a2+pt_a3);
	const float pt_e1 = def_we*(pt_a1+pt_a2+pt_a3);
	const float pt_e2 = def_we*(pt_a1+pt_a2-pt_a3);
	const float pt_e3 = def_we*(pt_a1-2.0f*pt_a2);
	// ★ 11.09.2026 (E5): Gatter von t%100 auf t%1000. Der Block kostete 0,33 % Wanduhr
	// (-18,9 s bei 4 mm, A/B-belegt). Die Aussagekraft bleibt: bei 4 mm sind das noch 50
	// Stichprobenschritte statt 501, und die Abnahme pruefe_ptrt vergleicht ohnehin nur
	// relativ (203 gegen 202) und auf Ungleichnull -- sie rechnet KEINE Sollzahl aus dem
	// Raster. Der Saettigungsvorbehalt fuer 199..201 bleibt unveraendert gueltig.
	if(t%1000ul==0ul) {
		// Drei Zaehler statt einem: einer allein bewiese nur, dass der Block betreten wurde.
		// Muster wie am NUT_SKAL-Diskriminator. Saettigend, sonst wickeln sie binnen Sekunden.
		if(rho_clamp_hits[199]<0xF0000000u) atomic_inc(&rho_clamp_hits[199]); // Wirkpfad: Block besucht
		const float pt_mabs = fabs(pt_m1)+fabs(pt_m2)+fabs(pt_m3);
		if(pt_mabs>0.0f&&rho_clamp_hits[200]<0xF0000000u) atomic_inc(&rho_clamp_hits[200]); // es gab ueberhaupt Geistanteil
		const float pt_dabs = fabs(pt_d0)+fabs(pt_dx)+fabs(pt_dy)+fabs(pt_dz)+fabs(pt_e1)+fabs(pt_e2)+fabs(pt_e3);
		if(pt_dabs>0.0f&&rho_clamp_hits[201]<0xF0000000u) atomic_inc(&rho_clamp_hits[201]); // der Abzug ist WIRKLICH ungleich null
		// ★★ ZWEITZAEHLUNG, ausgeduennt ueber den Zellindex (Kernel- und Host-Pruefer 10.09. nachts).
		// ZWEI Gruende, und beide sind hart:
		//  (1) 199..201 SAETTIGEN bei 4 mm nach acht Stichproben, also nach 800 von 50.100 Schritten.
		//      Schlimmer: EIN Stichprobenschritt (446 Mio Fluidzellen) uebersteigt den Kopfraum
		//      zwischen 0xF0000000 und 2^32 um Faktor 1,7 -- der Zaehler kann WICKELN und dann
		//      unter der Saettigungsschranke landen, worauf der Host faelschlich vergleicht.
		//      Mit n%1024 bleiben ~507.000 Zaehlungen je Stichprobe, also keine Saettigung im Lauf.
		//  (2) 200/201 KOENNEN GAR NICHT DURCHFALLEN. Ein reiner Chapman-Enskog-Zustand hat
		//      Geistmoment 3,1e-18, nach EINEM FP16S-Speicherumlauf steht dort 1,08e-6. Jede Zelle
		//      meldet also Geistanteil, und 201 == 200 ist eine Tautologie. Slot 203 prueft
		//      stattdessen, ob der Abzug die SPEICHERRUNDUNG ueberlebt: relative FP16S-Aufloesung
		//      9,8e-4 auf der Stoerform f^ = f - w_i. BERICHTIGT 22.09.2026: hier stand die Angabe
		//      gemessen, AUDIT-BEFUNDE B77 -- das ist eine falsche Herkunftsangabe. B77 misst etwas anderes
		//      [AUDIT-BEFUNDE.md:5499-5520: cz_druck_rest unter FP16C, -0,0119 bei 2,17 sigma]; die
		//      9,8e-4 sind dort hergeleitet, nicht gemessen. Rechnerisch ist FP16S die IEEE-half
		//      1-5-10 [lbm.cpp:2409-2413, defines.hpp:27; FP16C ist auskommentiert, defines.hpp:80],
		//      also 10 Mantissenbits: volle ULP 2^-10 = 9,8e-4, halbe ULP = echter Worst-Case-
		//      Rundungsfehler = 2^-11 = 4,9e-4. Die Schwelle hier ist damit um Faktor 2 zu scharf,
		//      Slot 203 UNTERTREIBT also. Sie bleibt trotzdem stehen: eine Aenderung wuerde die
		//      Vergleichbarkeit zu allen bisherigen Laeufen brechen, ohne etwas zu entscheiden.
		//      Die Gegenstelle kernel.cpp:5107-5109 nennt richtig 2^-11 = 4,9e-4. Bleibt 203
		//      nahe null, ist der Abzug kleiner als das Speicherquantum und wirkungslos --
		//      genau die Regression, die 199..201 nicht sehen.
		if(n%1024u==0u) {
			if(rho_clamp_hits[202]<0xF0000000u) atomic_inc(&rho_clamp_hits[202]);
			if(fabs(pt_e1)>9.8e-4f*fabs(fhn[7])&&rho_clamp_hits[203]<0xF0000000u) atomic_inc(&rho_clamp_hits[203]);
		}
	}
)+"#endif"+R( // PTRT
)+"#if defined(SRT)"+R(
)+"#ifdef VOLUME_FORCE"+R(
	const float c_tau = fma(w, -0.5f, 1.0f);
	for(uint i=0u; i<def_velocity_set; i++) Fin[i] *= c_tau;
)+"#endif"+R( // VOLUME_FORCE
)+"#ifndef EQUILIBRIUM_BOUNDARIES"+R(
	for(uint i=0u; i<def_velocity_set; i++) fhn[i] = fma(1.0f-w, fhn[i], fma(w, feq[i], Fin[i])); // perform collision (SRT)
)+"#ifdef PTRT"+R(
	fhn[0] += pt_d0;
	fhn[1] += pt_dx; fhn[2] += pt_dx;
	fhn[3] += pt_dy; fhn[4] += pt_dy;
	fhn[5] += pt_dz; fhn[6] += pt_dz;
	fhn[7] += pt_e1; fhn[8] += pt_e1; fhn[13] += pt_e1; fhn[14] += pt_e1;
	fhn[9] += pt_e2; fhn[10] += pt_e2; fhn[15] += pt_e2; fhn[16] += pt_e2;
	fhn[11] += pt_e3; fhn[12] += pt_e3; fhn[17] += pt_e3; fhn[18] += pt_e3;
)+"#endif"+R( // PTRT
)+"#else"+R( // EQUILIBRIUM_BOUNDARIES
	// ★ TYPE_E-Zweig HERAUSGEHOBEN statt als Ternaer in der Schleife: der fruehere Aufbau expandierte
	// den Randausdruck 19-fach in die Kollisionszeile, daran blieb der Uebersetzer haengen. So sieht
	// er zwei kleine, getrennte Schleifen. Nebeneffekt (Pruefer-Befund): die f_neq-Arbeit laeuft nur
	// noch fuer Randzellen, vorher rechnete JEDE Zelle die 19 Terme unbedingt.
	if(flagsn_bo==TYPE_E) {
		for(uint i=0u; i<def_velocity_set; i++) fhn[i] = REG_E(i); // f_eq bzw. f_eq + f_neq (regularisiert)
	} else {
		for(uint i=0u; i<def_velocity_set; i++) fhn[i] = fma(1.0f-w, fhn[i], fma(w, feq[i], Fin[i])); // perform collision (SRT)
)+"#ifdef PTRT"+R(
		fhn[0] += pt_d0;
		fhn[1] += pt_dx; fhn[2] += pt_dx;
		fhn[3] += pt_dy; fhn[4] += pt_dy;
		fhn[5] += pt_dz; fhn[6] += pt_dz;
		fhn[7] += pt_e1; fhn[8] += pt_e1; fhn[13] += pt_e1; fhn[14] += pt_e1;
		fhn[9] += pt_e2; fhn[10] += pt_e2; fhn[15] += pt_e2; fhn[16] += pt_e2;
		fhn[11] += pt_e3; fhn[12] += pt_e3; fhn[17] += pt_e3; fhn[18] += pt_e3;
)+"#endif"+R( // PTRT
	}
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
)+"#ifdef POSITIV"+R(
	{ // ★ 15.09.2026 Klemmen Stufe 1 P1b (KLEMMEN-STUFE1-PLAN.md §1-§4): Positivitaetsbegrenzer in PROJEKTIONSFORM, hier der Messarm.
	  // f* = fhn + w_i nach Kollision und P-TRT; g = f* - f_eq; G_i = g_i - w_i (m0 + 3 c_i.m) traegt weder Masse noch Impuls,
	  // f** = f* - (1-s) G_i erhaelt beide fuer JEDES s. s = kleinster Faktor, der alle f**_i >= tau_i haelt (pos_s).
	  // Nachtrag P1b (Absturzsperre 15.09.): alle Zellzaehler nur an Zaehlschritten t%def_zaehl_takt == 2 -- die natuerliche Rate
	  // negativer Populationen ist ungemessen, ungegatete Atomics in vielen Zellen je Schritt haben die B70 schon einmal lahmgelegt.
	  const bool pz_ = t%def_zaehl_takt==2ul&&n%(uxx)def_pos_sub==0u; // Zaehlschritt UND Stichprobenzelle (def_pos_sub aus dem belegt sicheren Gitter)
	  if(pneg_) { if(t%def_zaehl_takt==2ul&&n%(uxx)def_pos_sub==0u&&rho_clamp_hits[285]<0xF0000000u) atomic_inc(&rho_clamp_hits[285]); if(t==(ulong)def_zaehl_takt+3ul&&rho_clamp_hits[289]<0xF0000000u) atomic_inc(&rho_clamp_hits[289]); }
	  if(t==(ulong)def_zaehl_takt+2ul&&rho_clamp_hits[271]<0xF0000000u) atomic_inc(&rho_clamp_hits[271]); // Besuche am Pruefpunkt (Soll: Host-Flagzaehlung)
)+"#ifdef POSITIV_HAKEN1"+R(
	  bool ph_ = false; // Haken 1: masse- und impulsfreie Stoerung a = 2,5 w_s an reinen Fluidzellen ausserhalb der Randschale, Soll s zwischen 0,25 und 0,5
	  if(t==(ulong)def_zaehl_takt+2ul&&flagsn_bo==0u&&n%def_pos_hP==0u&&!klemm_randschale(n)) { fhn[0] += 5.0f*def_ws; fhn[1] -= 2.5f*def_ws; fhn[2] -= 2.5f*def_ws; ph_ = true; }
)+"#endif"+R( // POSITIV_HAKEN1
)+"#ifdef POSITIV_ANWENDEN"+R(
	  const bool pk_ = (fhn[0]<def_pos_g0)|(fhn[1]<def_pos_gs)|(fhn[2]<def_pos_gs)|(fhn[3]<def_pos_gs)|(fhn[4]<def_pos_gs)|(fhn[5]<def_pos_gs)|(fhn[6]<def_pos_gs)|(fhn[7]<def_pos_ge)|(fhn[8]<def_pos_ge)|(fhn[9]<def_pos_ge)|(fhn[10]<def_pos_ge)|(fhn[11]<def_pos_ge)|(fhn[12]<def_pos_ge)|(fhn[13]<def_pos_ge)|(fhn[14]<def_pos_ge)|(fhn[15]<def_pos_ge)|(fhn[16]<def_pos_ge)|(fhn[17]<def_pos_ge)|(fhn[18]<def_pos_ge);
)+"#else"+R(
	  const bool pk_ = pz_&&((fhn[0]<def_pos_g0)|(fhn[1]<def_pos_gs)|(fhn[2]<def_pos_gs)|(fhn[3]<def_pos_gs)|(fhn[4]<def_pos_gs)|(fhn[5]<def_pos_gs)|(fhn[6]<def_pos_gs)|(fhn[7]<def_pos_ge)|(fhn[8]<def_pos_ge)|(fhn[9]<def_pos_ge)|(fhn[10]<def_pos_ge)|(fhn[11]<def_pos_ge)|(fhn[12]<def_pos_ge)|(fhn[13]<def_pos_ge)|(fhn[14]<def_pos_ge)|(fhn[15]<def_pos_ge)|(fhn[16]<def_pos_ge)|(fhn[17]<def_pos_ge)|(fhn[18]<def_pos_ge)); // Modus 1: nur an Zaehlschritten pruefen -- die Felder bleiben ohnehin unberuehrt
)+"#endif"+R( // POSITIV_ANWENDEN
	  if(pk_) {
		if(flagsn_bo==TYPE_E) { if(pz_&&rho_clamp_hits[291]<0xF0000000u) atomic_inc(&rho_clamp_hits[291]); } // TYPE_E: f = f_eq, begrenzen sinnlos -- nur zaehlen
		else {
			const float pm0_ = (fhn[0]-feq[0])+(fhn[1]-feq[1])+(fhn[2]-feq[2])+(fhn[3]-feq[3])+(fhn[4]-feq[4])+(fhn[5]-feq[5])+(fhn[6]-feq[6])+(fhn[7]-feq[7])+(fhn[8]-feq[8])+(fhn[9]-feq[9])+(fhn[10]-feq[10])+(fhn[11]-feq[11])+(fhn[12]-feq[12])+(fhn[13]-feq[13])+(fhn[14]-feq[14])+(fhn[15]-feq[15])+(fhn[16]-feq[16])+(fhn[17]-feq[17])+(fhn[18]-feq[18]);
			const float pmx_ = (fhn[1]-feq[1])-(fhn[2]-feq[2])+(fhn[7]-feq[7])-(fhn[8]-feq[8])+(fhn[9]-feq[9])-(fhn[10]-feq[10])+(fhn[13]-feq[13])-(fhn[14]-feq[14])+(fhn[15]-feq[15])-(fhn[16]-feq[16]);
			const float pmy_ = (fhn[3]-feq[3])-(fhn[4]-feq[4])+(fhn[7]-feq[7])-(fhn[8]-feq[8])+(fhn[11]-feq[11])-(fhn[12]-feq[12])-(fhn[13]-feq[13])+(fhn[14]-feq[14])+(fhn[17]-feq[17])-(fhn[18]-feq[18]);
			const float pmz_ = (fhn[5]-feq[5])-(fhn[6]-feq[6])+(fhn[9]-feq[9])-(fhn[10]-feq[10])+(fhn[11]-feq[11])-(fhn[12]-feq[12])-(fhn[15]-feq[15])+(fhn[16]-feq[16])-(fhn[17]-feq[17])+(fhn[18]-feq[18]);
			float ps_ = 1.0f;
			ps_ = fmin(ps_, pos_s(fhn[0], feq[0], def_w0, def_pos_g0, pm0_));
			ps_ = fmin(ps_, pos_s(fhn[1], feq[1], def_ws, def_pos_gs, pm0_+3.0f*(pmx_)));
			ps_ = fmin(ps_, pos_s(fhn[2], feq[2], def_ws, def_pos_gs, pm0_+3.0f*(-pmx_)));
			ps_ = fmin(ps_, pos_s(fhn[3], feq[3], def_ws, def_pos_gs, pm0_+3.0f*(pmy_)));
			ps_ = fmin(ps_, pos_s(fhn[4], feq[4], def_ws, def_pos_gs, pm0_+3.0f*(-pmy_)));
			ps_ = fmin(ps_, pos_s(fhn[5], feq[5], def_ws, def_pos_gs, pm0_+3.0f*(pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[6], feq[6], def_ws, def_pos_gs, pm0_+3.0f*(-pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[7], feq[7], def_we, def_pos_ge, pm0_+3.0f*(pmx_+pmy_)));
			ps_ = fmin(ps_, pos_s(fhn[8], feq[8], def_we, def_pos_ge, pm0_+3.0f*(-pmx_-pmy_)));
			ps_ = fmin(ps_, pos_s(fhn[9], feq[9], def_we, def_pos_ge, pm0_+3.0f*(pmx_+pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[10], feq[10], def_we, def_pos_ge, pm0_+3.0f*(-pmx_-pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[11], feq[11], def_we, def_pos_ge, pm0_+3.0f*(pmy_+pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[12], feq[12], def_we, def_pos_ge, pm0_+3.0f*(-pmy_-pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[13], feq[13], def_we, def_pos_ge, pm0_+3.0f*(pmx_-pmy_)));
			ps_ = fmin(ps_, pos_s(fhn[14], feq[14], def_we, def_pos_ge, pm0_+3.0f*(-pmx_+pmy_)));
			ps_ = fmin(ps_, pos_s(fhn[15], feq[15], def_we, def_pos_ge, pm0_+3.0f*(pmx_-pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[16], feq[16], def_we, def_pos_ge, pm0_+3.0f*(-pmx_+pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[17], feq[17], def_we, def_pos_ge, pm0_+3.0f*(pmy_-pmz_)));
			ps_ = fmin(ps_, pos_s(fhn[18], feq[18], def_we, def_pos_ge, pm0_+3.0f*(-pmy_+pmz_)));
			if(pz_) {
				if(rho_clamp_hits[272]<0xF0000000u) atomic_inc(&rho_clamp_hits[272]); // Kandidat (Nicht-E)
				if((kl&3u)!=0u&&rho_clamp_hits[286]<0xF0000000u) atomic_inc(&rho_clamp_hits[286]); // Koinzidenz Dichteklemme
				if((kl&4u)!=0u&&rho_clamp_hits[287]<0xF0000000u) atomic_inc(&rho_clamp_hits[287]); // Koinzidenz u-Klemme
			}
			if(ps_<0.0f) { // machtlos (E3): f* bleibt, nur zaehlen
				if(pz_) { if(rho_clamp_hits[278]<0xF0000000u) atomic_inc(&rho_clamp_hits[278]); if(ps_<-1.5f&&rho_clamp_hits[279]<0xF0000000u) atomic_inc(&rho_clamp_hits[279]); }
			} else {
				uint pkl_ = 4u; bool pfb_ = false; // Klassen K0..K4 wie Stufe 0 (koordinatenbasiert, keine Nachbarsuche)
)+"#ifdef FORCE_FIELD"+R(
				{ uxx pfbk_; pfb_ = f_bbox(n, &pfbk_);
)+"#ifdef FACETTEN"+R(
				  if(pfb_&&flagsn_bo!=TYPE_MS) { if(fac_fid(fac_idx, pfbk_)!=0xFFFFFFFFu) pkl_ = 0u; }
)+"#endif"+R( // FACETTEN
				}
)+"#endif"+R( // FORCE_FIELD
				if(pkl_==4u) { if(flagsn_bo==TYPE_MS) pkl_ = 1u; else if(pfb_) pkl_ = 2u; else if(klemm_randschale(n)) pkl_ = 3u; }
				if(pz_) {
)+"#ifdef POSITIV_HAKEN3"+R(
					if(n%7u!=0u)
)+"#endif"+R( // POSITIV_HAKEN3
					{ if(rho_clamp_hits[273u+pkl_]<0xF0000000u) atomic_inc(&rho_clamp_hits[273u+pkl_]); }
					const uint pb_ = ps_<0.25f ? 0u : (ps_<0.5f ? 1u : (ps_<0.75f ? 2u : (ps_<0.95f ? 3u : 4u)));
					if(rho_clamp_hits[280u+pb_]<0xF0000000u) atomic_inc(&rho_clamp_hits[280u+pb_]);
)+"#ifdef POSITIV_HAKEN1"+R(
					if(ph_&&pb_==1u&&rho_clamp_hits[288]<0xF0000000u) atomic_inc(&rho_clamp_hits[288]);
)+"#endif"+R( // POSITIV_HAKEN1
				}
)+"#ifdef POSITIV_ANWENDEN"+R(
				// ★ 15.09.2026 Klemmen Stufe 1 P1c: ANWENDEN. f** = f* - (1-s) G_i erhaelt Masse und Impuls fuer jedes s (Projektionsform, E1).
				// E2: Facettenzellen K0 ausgenommen (Wandschub-Weitergabe), ausser CFD_POSITIV_FACETTE=1. s = 1 (Rundung) aendert nichts.
				bool pan_ = ps_<1.0f;
)+"#ifndef POSITIV_FACETTE"+R(
				pan_ = pan_&&pkl_!=0u;
)+"#endif"+R( // POSITIV_FACETTE
)+"#ifdef POSITIV_FACETTE"+R(
				// ★ Audit-Schleife 16.09.2026, Befund B3: CFD_POSITIV_FACETTE aenderte nur die Bedingung oben und hatte KEINEN eigenen Zaehler --
				// [273] zaehlt K0-Kandidaten in BEIDEN Stellungen gleich, der Schalter war also nicht vom Nicht-Schalter unterscheidbar.
				// Slot 303 zaehlt genau das, was der Schalter zusaetzlich zulaesst: eine K0-Zelle (Facette), die WIRKLICH begrenzt wird.
				// Soll unter CFD_POSITIV_FACETTE=1: > 0, sobald [273] > 0 (K0-Kandidaten in der Stichprobe vorhanden).
				// GATTERUNG wie [273]/[280]: NUR an Zaehlschritten und Stichprobenzellen (pz_). Ein ungegateter Zaehler im Immerpfad hat am
				// 15.09. die B70 lahmgelegt (device wedged) -- der Zaehler ist ein Wirkpfadbeleg, kein Mengenmass.
				if(pz_&&pan_&&pkl_==0u&&rho_clamp_hits[303]<0xF0000000u) atomic_inc(&rho_clamp_hits[303]);
)+"#endif"+R( // POSITIV_FACETTE
				if(pan_) {
					const float pq_ = 1.0f-ps_;
					float pab_ = 0.0f; // Summe |d_i| (Wirkungsgroesse)
)+"#ifdef POSITIV_HAKEN1"+R(
					float pdm_ = 0.0f, pdx_ = 0.0f, pdy_ = 0.0f, pdz_ = 0.0f; // Selbstpruefung: Masse und Impuls der Korrektur
					{ const float d_ = pq_*((fhn[0]-feq[0])-def_w0*(pm0_)); fhn[0] -= d_; pab_ += fabs(d_); pdm_ += d_; }
					{ const float d_ = pq_*((fhn[1]-feq[1])-def_ws*(pm0_+3.0f*(pmx_))); fhn[1] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ += d_; }
					{ const float d_ = pq_*((fhn[2]-feq[2])-def_ws*(pm0_+3.0f*(-pmx_))); fhn[2] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ -= d_; }
					{ const float d_ = pq_*((fhn[3]-feq[3])-def_ws*(pm0_+3.0f*(pmy_))); fhn[3] -= d_; pab_ += fabs(d_); pdm_ += d_; pdy_ += d_; }
					{ const float d_ = pq_*((fhn[4]-feq[4])-def_ws*(pm0_+3.0f*(-pmy_))); fhn[4] -= d_; pab_ += fabs(d_); pdm_ += d_; pdy_ -= d_; }
					{ const float d_ = pq_*((fhn[5]-feq[5])-def_ws*(pm0_+3.0f*(pmz_))); fhn[5] -= d_; pab_ += fabs(d_); pdm_ += d_; pdz_ += d_; }
					{ const float d_ = pq_*((fhn[6]-feq[6])-def_ws*(pm0_+3.0f*(-pmz_))); fhn[6] -= d_; pab_ += fabs(d_); pdm_ += d_; pdz_ -= d_; }
					{ const float d_ = pq_*((fhn[7]-feq[7])-def_we*(pm0_+3.0f*(pmx_+pmy_))); fhn[7] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ += d_; pdy_ += d_; }
					{ const float d_ = pq_*((fhn[8]-feq[8])-def_we*(pm0_+3.0f*(-pmx_-pmy_))); fhn[8] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ -= d_; pdy_ -= d_; }
					{ const float d_ = pq_*((fhn[9]-feq[9])-def_we*(pm0_+3.0f*(pmx_+pmz_))); fhn[9] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ += d_; pdz_ += d_; }
					{ const float d_ = pq_*((fhn[10]-feq[10])-def_we*(pm0_+3.0f*(-pmx_-pmz_))); fhn[10] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ -= d_; pdz_ -= d_; }
					{ const float d_ = pq_*((fhn[11]-feq[11])-def_we*(pm0_+3.0f*(pmy_+pmz_))); fhn[11] -= d_; pab_ += fabs(d_); pdm_ += d_; pdy_ += d_; pdz_ += d_; }
					{ const float d_ = pq_*((fhn[12]-feq[12])-def_we*(pm0_+3.0f*(-pmy_-pmz_))); fhn[12] -= d_; pab_ += fabs(d_); pdm_ += d_; pdy_ -= d_; pdz_ -= d_; }
					{ const float d_ = pq_*((fhn[13]-feq[13])-def_we*(pm0_+3.0f*(pmx_-pmy_))); fhn[13] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ += d_; pdy_ -= d_; }
					{ const float d_ = pq_*((fhn[14]-feq[14])-def_we*(pm0_+3.0f*(-pmx_+pmy_))); fhn[14] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ -= d_; pdy_ += d_; }
					{ const float d_ = pq_*((fhn[15]-feq[15])-def_we*(pm0_+3.0f*(pmx_-pmz_))); fhn[15] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ += d_; pdz_ -= d_; }
					{ const float d_ = pq_*((fhn[16]-feq[16])-def_we*(pm0_+3.0f*(-pmx_+pmz_))); fhn[16] -= d_; pab_ += fabs(d_); pdm_ += d_; pdx_ -= d_; pdz_ += d_; }
					{ const float d_ = pq_*((fhn[17]-feq[17])-def_we*(pm0_+3.0f*(pmy_-pmz_))); fhn[17] -= d_; pab_ += fabs(d_); pdm_ += d_; pdy_ += d_; pdz_ -= d_; }
					{ const float d_ = pq_*((fhn[18]-feq[18])-def_we*(pm0_+3.0f*(-pmy_+pmz_))); fhn[18] -= d_; pab_ += fabs(d_); pdm_ += d_; pdy_ -= d_; pdz_ += d_; }
					if(pz_&&(fabs(pdm_)>1.0E-5f*pab_+1.0E-9f||fabs(pdx_)>1.0E-5f*pab_+1.0E-9f||fabs(pdy_)>1.0E-5f*pab_+1.0E-9f||fabs(pdz_)>1.0E-5f*pab_+1.0E-9f)&&rho_clamp_hits[290]<0xF0000000u) atomic_inc(&rho_clamp_hits[290]); // Soll 0: Toleranz 1e-5 der Korrekturgroesse (float32-Summe ueber 19 Terme)
)+"#else"+R(
					{ const float d_ = pq_*((fhn[0]-feq[0])-def_w0*(pm0_)); fhn[0] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[1]-feq[1])-def_ws*(pm0_+3.0f*(pmx_))); fhn[1] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[2]-feq[2])-def_ws*(pm0_+3.0f*(-pmx_))); fhn[2] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[3]-feq[3])-def_ws*(pm0_+3.0f*(pmy_))); fhn[3] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[4]-feq[4])-def_ws*(pm0_+3.0f*(-pmy_))); fhn[4] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[5]-feq[5])-def_ws*(pm0_+3.0f*(pmz_))); fhn[5] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[6]-feq[6])-def_ws*(pm0_+3.0f*(-pmz_))); fhn[6] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[7]-feq[7])-def_we*(pm0_+3.0f*(pmx_+pmy_))); fhn[7] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[8]-feq[8])-def_we*(pm0_+3.0f*(-pmx_-pmy_))); fhn[8] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[9]-feq[9])-def_we*(pm0_+3.0f*(pmx_+pmz_))); fhn[9] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[10]-feq[10])-def_we*(pm0_+3.0f*(-pmx_-pmz_))); fhn[10] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[11]-feq[11])-def_we*(pm0_+3.0f*(pmy_+pmz_))); fhn[11] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[12]-feq[12])-def_we*(pm0_+3.0f*(-pmy_-pmz_))); fhn[12] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[13]-feq[13])-def_we*(pm0_+3.0f*(pmx_-pmy_))); fhn[13] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[14]-feq[14])-def_we*(pm0_+3.0f*(-pmx_+pmy_))); fhn[14] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[15]-feq[15])-def_we*(pm0_+3.0f*(pmx_-pmz_))); fhn[15] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[16]-feq[16])-def_we*(pm0_+3.0f*(-pmx_+pmz_))); fhn[16] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[17]-feq[17])-def_we*(pm0_+3.0f*(pmy_-pmz_))); fhn[17] -= d_; pab_ += fabs(d_); }
					{ const float d_ = pq_*((fhn[18]-feq[18])-def_we*(pm0_+3.0f*(-pmy_+pmz_))); fhn[18] -= d_; pab_ += fabs(d_); }
)+"#endif"+R( // POSITIV_HAKEN1
					if(pz_) { pos_summe(rho_clamp_hits, 292u, pq_); pos_summe(rho_clamp_hits, 293u, pab_); } // Festkomma-Summen (1-s) und Sum|d| an Zaehlschritten und Stichprobenzellen
				}
)+"#endif"+R( // POSITIV_ANWENDEN
			}
		}
	  }
	}
)+"#endif"+R( // POSITIV
)+"#elif defined(TRT)"+R(
	const float wp = w; // TRT: inverse of "+" relaxation time
	const float wm = 1.0f/(def_lambda/(1.0f/w-0.5f)+0.5f); // TRT: inverse of "-" relaxation time wm = 1.0f/(0.1875f/(3.0f*nu)+0.5f), nu = (1.0f/w-0.5f)/3.0f;
)+"#ifdef VOLUME_FORCE"+R(
	const float c_taup=fma(wp, -0.25f, 0.5f), c_taum=fma(wm, -0.25f, 0.5f); // source: https://arxiv.org/pdf/1901.08766.pdf
	float Fib[def_velocity_set]; // F_bar
	Fib[0] = Fin[0];
	for(uint i=1u; i<def_velocity_set; i+=2u) {
		Fib[i   ] = Fin[i+1u];
		Fib[i+1u] = Fin[i   ];
	}
	for(uint i=0u; i<def_velocity_set; i++) Fin[i] = fma(c_taup, Fin[i]+Fib[i], c_taum*(Fin[i]-Fib[i]));
)+"#endif"+R( // VOLUME_FORCE
	float fhb[def_velocity_set]; // fhn in inverse directions
	float feb[def_velocity_set]; // feq in inverse directions
	fhb[0] = fhn[0];
	feb[0] = feq[0];
	for(uint i=1u; i<def_velocity_set; i+=2u) {
		fhb[i   ] = fhn[i+1u];
		fhb[i+1u] = fhn[i   ];
		feb[i   ] = feq[i+1u];
		feb[i+1u] = feq[i   ];
	}
)+"#ifndef EQUILIBRIUM_BOUNDARIES"+R(
	for(uint i=0u; i<def_velocity_set; i++) fhn[i] = fma(0.5f*wp, feq[i]-fhn[i]+feb[i]-fhb[i], fma(0.5f*wm, feq[i]-feb[i]-fhn[i]+fhb[i], fhn[i]+Fin[i])); // perform collision (TRT)
)+"#ifdef PTRT"+R(
		fhn[0] += pt_d0;
	fhn[1] += pt_dx; fhn[2] += pt_dx;
	fhn[3] += pt_dy; fhn[4] += pt_dy;
	fhn[5] += pt_dz; fhn[6] += pt_dz;
	fhn[7] += pt_e1; fhn[8] += pt_e1; fhn[13] += pt_e1; fhn[14] += pt_e1;
	fhn[9] += pt_e2; fhn[10] += pt_e2; fhn[15] += pt_e2; fhn[16] += pt_e2;
	fhn[11] += pt_e3; fhn[12] += pt_e3; fhn[17] += pt_e3; fhn[18] += pt_e3;
)+"#endif"+R( // PTRT
)+"#else"+R( // EQUILIBRIUM_BOUNDARIES
	// ★ TYPE_E-Zweig herausgehoben -- Begruendung wie im SRT-Zweig.
	if(flagsn_bo==TYPE_E) {
		for(uint i=0u; i<def_velocity_set; i++) fhn[i] = REG_E(i); // f_eq bzw. f_eq + f_neq (regularisiert)
	} else {
		for(uint i=0u; i<def_velocity_set; i++) fhn[i] = fma(0.5f*wp, feq[i]-fhn[i]+feb[i]-fhb[i], fma(0.5f*wm, feq[i]-feb[i]-fhn[i]+fhb[i], fhn[i]+Fin[i])); // perform collision (TRT)
)+"#ifdef PTRT"+R(
			fhn[0] += pt_d0;
		fhn[1] += pt_dx; fhn[2] += pt_dx;
		fhn[3] += pt_dy; fhn[4] += pt_dy;
		fhn[5] += pt_dz; fhn[6] += pt_dz;
		fhn[7] += pt_e1; fhn[8] += pt_e1; fhn[13] += pt_e1; fhn[14] += pt_e1;
		fhn[9] += pt_e2; fhn[10] += pt_e2; fhn[15] += pt_e2; fhn[16] += pt_e2;
		fhn[11] += pt_e3; fhn[12] += pt_e3; fhn[17] += pt_e3; fhn[18] += pt_e3;
)+"#endif"+R( // PTRT
	}
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
)+"#endif"+R( // TRT

	// ★ Rang-1-Remat (Perf-Audit Achse 2, 2026-08-26): zur Laufzeit ist t>>62 == 0 (t = monotoner
	// Schrittzaehler < 2^62, lbm.hpp/lbm.cpp -- diese Semantik ist TRAGEND, nie Bits in t packen!),
	// also nn == n und j2 == j BITGENAU. IGC kann das nicht beweisen und rechnet die 19 fi-Adressen
	// hier neu, statt sie ueber den Facettenblock zu spillen (Spill 448/832 -> 0/0, offline bewiesen,
	// FP-Instruktions-Multiset identisch; Belegkette: AUDIT-BEFUNDE Rang-1-Absatz + Abnahme g24).
	const uxx nn = n+(uxx)(t>>62);
	uxx j2[def_velocity_set]; // rematerialisierte Nachbarindizes, identische Werte wie j
	neighbors(nn, j2);
	store_f(nn, fhn, fi, j2, t TS_A); // perform streaming (part 1)
} // stream_collide()

)+"#ifdef SURFACE"+R(
)+R(kernel void surface_0(global fpxx* fi, const global float* rho, const global float* u, const global uchar* flags, global float* mass, const global float* massex, const global float* phi, const ulong t, const float fx, const float fy, const float fz) { // capture outgoing DDFs before streaming
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute surface_0() on halo
	const uchar flagsn = flags[n]; // cache flags[n] for multiple readings
	const uchar flagsn_bo=flagsn&TYPE_BO, flagsn_su=flagsn&TYPE_SU; // extract boundary and surface flags
	if(flagsn_bo==TYPE_S||flagsn_su==TYPE_G) return; // cell processed here is fluid or interface

	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	float fhn[def_velocity_set]; // incoming DDFs
	load_f(n, fhn, fi, j, t); // load incoming DDFs
	float fon[def_velocity_set]; // outgoing DDFs
	fon[0] = fhn[0]; // fon[0] is already loaded in fhn[0]
	load_f_outgoing(n, fon, fi, j, t); // load outgoing DDFs

	float massn = mass[n];
	for(uint i=1u; i<def_velocity_set; i++) {
		massn += massex[j[i]]; // distribute excess mass from last step which is stored in neighbors
	}
	if(flagsn_su==TYPE_F) { // cell is fluid
		for(uint i=1u; i<def_velocity_set; i++) massn += fhn[i]-fon[i]; // neighbor is fluid or interface cell
	} else if(flagsn_su==TYPE_I) { // cell is interface
		float phij[def_velocity_set]; // cache fill level of neighbor lattice points
		for(uint i=1u; i<def_velocity_set; i++) phij[i] = phi[j[i]]; // cache fill level of neighbor lattice points
		float rhon, uxn, uyn, uzn, rho_laplace=0.0f; // no surface tension if rho_laplace is not overwritten later
)+"#ifndef EQUILIBRIUM_BOUNDARIES"+R(
		calculate_rho_u(fon, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fon (not fhn)
)+"#else"+R( // EQUILIBRIUM_BOUNDARIES
		if(flagsn_bo==TYPE_E) {
			rhon = rho[               n]; // apply preset velocity/density
			// ★ BEWUSST u[...] und NICHT load_u: die Signatur von surface_0 traegt weiter den
			// float-Zeigertyp (SURFACE ist nicht gebaut, defines.hpp sperrt die Kombination mit U_FP16
			// hart). Ein load_u darauf waere STILL falsch -- ocloc meldet dazu nur "incompatible
			// pointer types", und der Produktionsbau haengt -w an (opencl.hpp), die Warnung ist weg.
			// Vom Pruefagenten am 12.09. gefunden: erst stand hier load_u, die Signatur aber nicht.
			// ★ BERICHTIGT 12.09. abends (Pruefer C, und nachgemessen): hier stand, Kommentare in
			// R()-Bloecken landeten im emittierten Quelltext und haetten den Typ-Zensus auf 22
			// getrieben. DAS IST FALSCH. Der Praeprozessor entfernt Kommentare in Phase 3, die
			// Stringbildung ist Phase 4 -- mit demselben R()-Makro nachgestellt: ein Kommentar mit
			// dem Zeigertyp darin erscheint NICHT im Ergebnis. Die 22 kamen aus einem grep ueber die
			// QUELLDATEI, nicht aus dem emittierten Code; ich hatte sie falsch zugeordnet. Die echte
			// Falle an dieser Stelle ist die andere Richtung: ein "//" MITTEN in einer Kernelzeile
			// frisst den Code dahinter, weil er vor der Stringbildung verschwindet.
			uxn  = u[                 n];
			uyn  = u[    def_N+(ulong)n];
			uzn  = u[2ul*def_N+(ulong)n];
		} else {
			calculate_rho_u(fon, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fon (not fhn)
		}
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
		uxn = clamp(uxn, -def_c, def_c); // limit velocity (for stability purposes)
		uyn = clamp(uyn, -def_c, def_c);
		uzn = clamp(uzn, -def_c, def_c);
		phij[0] = calculate_phi(rhon, massn, flagsn); // don't load phi[n] from memory, instead recalculate it with mass corrected by excess mass
		rho_laplace = def_6_sigma==0.0f ? 0.0f : def_6_sigma*calculate_curvature(n, phij, phi); // surface tension least squares fit (PLIC, most accurate)
		float feg[def_velocity_set]; // reconstruct f from neighbor gas lattice points
		const float rho2tmp = 0.5f/rhon; // apply external volume force (Guo forcing, Krueger p.233f)
		const float uxntmp = clamp(fma(fx, rho2tmp, uxn), -def_c, def_c); // limit velocity (for stability purposes)
		const float uyntmp = clamp(fma(fy, rho2tmp, uyn), -def_c, def_c); // force term: F*dt/(2*rho)
		const float uzntmp = clamp(fma(fz, rho2tmp, uzn), -def_c, def_c);
		calculate_f_eq(1.0f-rho_laplace, uxntmp, uyntmp, uzntmp, feg); // calculate gas equilibrium DDFs with constant ambient pressure
		uchar flagsj_su[def_velocity_set]; // cache neighbor flags for multiple readings
		for(uint i=1u; i<def_velocity_set; i++) flagsj_su[i] = flags[j[i]]&TYPE_SU;
		for(uint i=1u; i<def_velocity_set; i+=2u) { // calculate mass exchange between current cell and fluid/interface cells
			massn += flagsj_su[i   ]&(TYPE_F|TYPE_I) ? flagsj_su[i   ]==TYPE_F ? fhn[i+1]-fon[i   ] : 0.5f*(phij[i   ]+phij[0])*(fhn[i+1 ]-fon[i   ]) : 0.0f; // neighbor is fluid or interface cell
			massn += flagsj_su[i+1u]&(TYPE_F|TYPE_I) ? flagsj_su[i+1u]==TYPE_F ? fhn[i  ]-fon[i+1u] : 0.5f*(phij[i+1u]+phij[0])*(fhn[i   ]-fon[i+1u]) : 0.0f; // fluid : interface : gas
		}
		for(uint i=1u; i<def_velocity_set; i+=2u) { // calculate reconstructed gas DDFs
			fhn[i   ] = feg[i+1u]-fon[i+1u]+feg[i   ];
			fhn[i+1u] = feg[i   ]-fon[i   ]+feg[i+1u];
		}
		store_f_reconstructed(n, fhn, fi, j, t, flagsj_su); // store reconstructed gas DDFs that are streamed in during the following stream_collide()
	}
	mass[n] = massn;
}
)+R(kernel void surface_1(global uchar* flags) { // prevent neighbors from interface->fluid cells to become/be gas cells
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N) return; // execute surface_1() also on halo
	const uchar flagsn_sus = flags[n]&(TYPE_SU|TYPE_S); // extract SURFACE flags
	if(flagsn_sus==TYPE_IF) { // flag interface->fluid is set
		uxx j[def_velocity_set]; // neighbor indices
		neighbors(n, j); // calculate neighbor indices
		for(uint i=1u; i<def_velocity_set; i++) {
			const uchar flagsji = flags[j[i]];
			const uchar flagsji_su = flagsji&(TYPE_SU|TYPE_S); // extract SURFACE flags
			const uchar flagsji_r = flagsji&~TYPE_SU; // extract all non-SURFACE flags
			if(flagsji_su==TYPE_IG) flags[j[i]] = flagsji_r|TYPE_I; // prevent interface neighbor cells from becoming gas
			else if(flagsji_su==TYPE_G) flags[j[i]] = flagsji_r|TYPE_GI; // neighbor cell was gas and must change to interface
		}
	}
} // possible types at the end of surface_1(): TYPE_F / TYPE_I / TYPE_G / TYPE_IF / TYPE_IG / TYPE_GI
)+R(kernel void surface_2(global fpxx* fi, const global float* rho, const global float* u, global uchar* flags, const ulong t) {  // apply flag changes and calculate excess mass
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N) return; // execute surface_2() also on halo
	const uchar flagsn_sus = flags[n]&(TYPE_SU|TYPE_S); // extract SURFACE flags
	if(flagsn_sus==TYPE_GI) { // initialize the fi of gas cells that should become interface
		float rhon, uxn, uyn, uzn; // average over all fluid/interface neighbors
		average_neighbors_non_gas(n, rho, u, flags, &rhon, &uxn, &uyn, &uzn); // get average rho/u from all fluid/interface neighbors
		float feq[def_velocity_set];
		calculate_f_eq(rhon, uxn, uyn, uzn, feq); // calculate equilibrium DDFs
		uxx j[def_velocity_set];
		neighbors(n, j);
		store_f(n, feq, fi, j, t); // write feq to fi in video memory
	} else if(flagsn_sus==TYPE_IG) { // flag interface->gas is set
		uxx j[def_velocity_set]; // neighbor indices
		neighbors(n, j); // calculate neighbor indices
		for(uint i=1u; i<def_velocity_set; i++) {
			const uchar flagsji = flags[j[i]];
			const uchar flagsji_su = flagsji&(TYPE_SU|TYPE_S); // extract SURFACE flags
			const uchar flagsji_r = flagsji&~TYPE_SU; // extract all non-SURFACE flags
			if(flagsji_su==TYPE_F||flagsji_su==TYPE_IF) {
				flags[j[i]] = flagsji_r|TYPE_I; // prevent fluid or interface neighbors that turn to fluid from being/becoming fluid
			}
		}
	}
} // possible types at the end of surface_2(): TYPE_F / TYPE_I / TYPE_G / TYPE_IF / TYPE_IG / TYPE_GI
)+R(kernel void surface_3(const global float* rho, global uchar* flags, global float* mass, global float* massex, global float* phi) { // apply flag changes and calculate excess mass
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute surface_3() on halo
	const uchar flagsn_sus = flags[n]&(TYPE_SU|TYPE_S); // extract SURFACE flags
	if(flagsn_sus&TYPE_S) return;
	const float rhon = rho[n]; // density of cell n
	float massn = mass[n]; // mass of cell n
	float massexn = 0.0f; // excess mass of cell n
	float phin = 0.0f;
	if(flagsn_sus==TYPE_F) { // regular fluid cell
		massexn = massn-rhon; // dump mass-rho difference into excess mass
		massn = rhon; // fluid cell mass has to equal rho
		phin = 1.0f;
	} else if(flagsn_sus==TYPE_I) { // regular interface cell
		massexn = massn>rhon ? massn-rhon : massn<0.0f ? massn : 0.0f; // allow interface cells with mass>rho or mass<0
		massn = clamp(massn, 0.0f, rhon);
		phin = calculate_phi(rhon, massn, TYPE_I); // calculate fill level for next step (only necessary for interface cells)
	} else if(flagsn_sus==TYPE_G) { // regular gas cell
		massexn = massn; // dump remaining mass into excess mass
		massn = 0.0f;
		phin = 0.0f;
	} else if(flagsn_sus==TYPE_IF) { // flag interface->fluid is set
		flags[n] = (flags[n]&~TYPE_SU)|TYPE_F; // cell becomes fluid
		massexn = massn-rhon; // dump mass-rho difference into excess mass
		massn = rhon; // fluid cell mass has to equal rho
		phin = 1.0f; // set phi[n] to 1.0f for fluid cells
	} else if(flagsn_sus==TYPE_IG) { // flag interface->gas is set
		flags[n] = (flags[n]&~TYPE_SU)|TYPE_G; // cell becomes gas
		massexn = massn; // dump remaining mass into excess mass
		massn = 0.0f; // gas mass has to be zero
		phin = 0.0f; // set phi[n] to 0.0f for gas cells
	} else if(flagsn_sus==TYPE_GI) { // flag gas->interface is set
		flags[n] = (flags[n]&~TYPE_SU)|TYPE_I; // cell becomes interface
		massexn = massn>rhon ? massn-rhon : massn<0.0f ? massn : 0.0f; // allow interface cells with mass>rho or mass<0
		massn = clamp(massn, 0.0f, rhon);
		phin = calculate_phi(rhon, massn, TYPE_I); // calculate fill level for next step (only necessary for interface cells)
	}
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	uint counter = 0u; // count (fluid|interface) neighbors
	for(uint i=1u; i<def_velocity_set; i++) { // simple model: distribute excess mass equally to all interface and fluid neighbors
		const uchar flagsji_su = flags[j[i]]&(TYPE_SU|TYPE_S); // extract SURFACE flags
		counter += (uint)(flagsji_su==TYPE_F||flagsji_su==TYPE_I||flagsji_su==TYPE_IF||flagsji_su==TYPE_GI); // avoid branching
	}
	massn += counter>0u ? 0.0f : massexn; // if excess mass can't be distributed to neighboring interface or fluid cells, add it to local mass (ensure mass conservation)
	massexn = counter>0u ? massexn/(float)counter : 0.0f; // divide excess mass up for all interface or fluid neighbors
	mass[n] = massn; // update mass
	massex[n] = massexn; // update excess mass
	phi[n] = phin; // update phi
} // possible types at the end of surface_3(): TYPE_F / TYPE_I / TYPE_G
)+"#endif"+R( // SURFACE
)
// ★ Audit-Befund 21 (latent, Kernel hat derzeit keine Aufrufer im Fork -- UPDATE_FIELDS liefert die
// Felder im stream_collide): dieser Kernel kennt WANDFUNKTION und FACETTEN NICHT. Wer ihn
// wiederbelebt, bekommt an behandelten Wandzellen rho/u aus den getauschten fi ohne Spiegel-/
// Facettenlogik -- erst nachziehen. (Host-Kommentar, bewusst NICHT emittiert: das
// CFD_DUMP_CL-Diff-Instrument soll kommentarfrei vergleichbar bleiben -- R3-Nachschliff.)
// ★ BODEN_EQ (Heiko 2026-08-20): V1-apply_floor_velocity-PORT gegen die Staggered-Mode der
// Moving-Floor-Injektion (Slice-Offensive P5/P6): post-stream Equilibrium-Reset mit LOKALEM
// rho (druckerhaltend) auf den Fluidzellen z=1..nz. TYPE_MS ist FLUID und wird BEHANDELT
// (V1-MS-Guard-Lehre 2f705ba; V1-Verifikation: Cz -31 %). Signatur V1-wortgetreu in EINEM
// R()-Block (Klammern balanciert -- Werkzeugfalle Variante 2).
+R(kernel void boden_eq(global fpxx* fi, const global uchar* flags, const ulong t, const float u_road, const uint nz, const uint nz_down, const uint x_split, const uint abstand, volatile global uint* diag TS_P) {
	const uxx n = get_global_id(0);
	if(n>=(uxx)def_N||is_halo(n)) return;
)+"#ifdef SPARSE_TILES"+R(
	if(is_dead_tile(n, tile_slot)) return; // XL-Audit B1
)+"#endif"+R( // SPARSE_TILES
	const uint3 xyz = coordinates(n);
	const uint nz_eff = (xyz.x>=x_split) ? nz_down : nz; // B4-Notiz: nz_down=0 heisst hier AUS ab Nase (V1: uniform nz) -- bewusste Abweichung (Kraefteschutz), V1-A/Bs mit nz_down=0 nicht bitvergleichbar // V1-x_split: ab Nase nz_down (Default 0 = unterm Wagen/Wake AUS), stromauf nz
	if(nz_eff==0u||!(xyz.z>=1u&&xyz.z<=nz_eff)) return; // Perf-Audit 2026-08-20: z-Test (reine Arithmetik) VOR dem flags-Load -- sonst streamt jede Enqueue das komplette flags-Feld (N x 1 B) fuer ein Band von <2 % der Zellen (V1-Erbdefekt)
	const uchar bo = flags[n]&TYPE_BO;
	if(bo==TYPE_S||bo==TYPE_E) return;
	if(abstand>0u) { // Heiko 2026-08-20: reifennahe Zellen NICHT aufpraegen (Kraefteschutz). Scan nur gleiche Ebene und OBERHALB (dz>=0) -- die Fahrbahn z=0 zaehlt bewusst nicht; damit werden AUCH diagonal-untere Solids ignoriert. Bei unterhalb der Achse konvexen Reifen faengt der In-Ebenen-Scan diese Faelle mit ab (geometrieabhaengige Annahme, XL-R2).
		const int a = (int)abstand; bool nah = false;
		for(int dz=0; dz<=a&&!nah; dz++) for(int dy=-a; dy<=a&&!nah; dy++) for(int dx=-a; dx<=a&&!nah; dx++) {
			const int xx=(int)xyz.x+dx, yy=(int)xyz.y+dy, zz=(int)xyz.z+dz;
			if(xx<0||yy<0||xx>=(int)def_Nx||yy>=(int)def_Ny||zz>=(int)def_Nz) continue;
			if((flags[index((uint3)((uint)xx,(uint)yy,(uint)zz))]&TYPE_BO)==TYPE_S) nah = true;
		}
		if(nah) { if(t%def_zaehl_takt==0ul) atomic_inc(&diag[117]); return; } // ★ S5: Aussparungen zaehlen (Slot 117) -- der Schalter CFD_BODEN_EQ_ABSTAND hatte keinen Wirkpfad
	}
	if(t%def_zaehl_takt==0ul) atomic_inc(&diag[20]); // XL-3 M2: Wirkpfad-Nachweis IM Binary (Iron Rule 3; die V1-Vorlage war jahrelang stiller No-Op)
	uxx j[def_velocity_set]; neighbors(n, j);
	float fhn[def_velocity_set]; load_f(n, fhn, fi, j, t TS_A);
	float rho_local, ux, uy, uz; calculate_rho_u(fhn, &rho_local, &ux, &uy, &uz); // XL-B8: post-stream-load = Paare vertauscht -> u waere NEGIERT (nur rho ist invariant und wird genutzt); XL-B7: coordinates() lokal -- Multi-Domain braeuchte def_O-Offsets (heute D=1)
)+"#if defined(KLEMM_BILANZ)&&defined(RHO_CLAMP)"+R(
	if(rho_local<=RHO_CLAMP_MIN||rho_local>=RHO_CLAMP_MAX) { // ★ 15.09.2026 Klemmen S0d (KLEMMEN-STUFE0-PLAN.md §1 Punkt 2): Faktor 1, weil f hier durch f_eq(rho_c) ERSETZT wird
		const float dq_ = fabs(rho_local-klemm_rho_roh(fhn));
		if(diag[266]<0xF0000000u) atomic_inc(&diag[266]);
		if((dq_>2.0f||(as_uint(dq_)&0x7F800000u)==0x7F800000u)&&diag[265]<0xF0000000u) atomic_inc(&diag[265]); // Kappung 2 (Host-Wickelschranke), Soll 0 -- ★ Audit 16.09.2026 (A-N6, H4): Exponent-Bittest, damit NaN/Inf gezaehlt wird
		atomic_add(&diag[rho_local<=RHO_CLAMP_MIN ? 267u : 268u], convert_uint_sat(fma(fmin(dq_, 2.0f), def_klemm_s, 0.5f)));
	}
)+"#endif"+R( // KLEMM_BILANZ
	float feq[def_velocity_set]; calculate_f_eq(rho_local, u_road, 0.0f, 0.0f, feq);
	store_f(n, feq, fi, j, t TS_A);
}
)
// ★ EINLASS_EQ (2026-08-21): V1-apply_inlet_velocity-PORT -- Freestream-Clamp der Spalten x=1..nx
// HINTER dem TYPE_E-Einlass (x=0 bleibt TYPE_E): post-stream Equilibrium-Reset mit LOKALEM rho
// (druckerhaltend), u=(u_road,0,0). Gegen die Einlass-Staggered-Streifen (einlass_saeule-Sonde).
// TYPE_MS ist FLUID und wird BEHANDELT (MS-Guard-Lehre 2f705ba -- V1s pauschaler TYPE_BO-Guard
// war ein Erbfehler). x-Test VOR dem flags-Load (Perf-Lehre boden_eq: sonst streamt jede Enqueue
// das komplette flags-Feld fuer <1 % der Zellen). UEBERLAPP mit boden_eq in den Ecken (x=1..nx,
// z=1..nz): beide schreiben dieselbe EQ-Form mit lokalem rho und u=(u_road,0,0) -- der zweite
// Kernel liest das rho des ersten (EQ-Momente sind rho-erhaltend), Reihenfolge also egal.
// Signatur in EINEM R()-Block (Klammerfalle, Werkzeugfalle Variante 2).
+R(// XL-B7-Latenz auch hier: coordinates() ist domaenenlokal -- bei D>1 mit x-Split klemmte jede
// Domaene ihr LOKALES x=1..nx mitten im Feld (x ist die uebliche Split-Achse!); heute D=1 ueberall.
// Sponge-Ueberlapp (x- gehoert zur Daempfungszone): harmlos -- einlass_eq verwirft post-stream
// alles ausser rho, die Sponge-Wirkung im Band ist damit wirkungslos (Pruefagent N2).
kernel void einlass_eq(global fpxx* fi, const global uchar* flags, const ulong t, const float u_road, const uint nx, volatile global uint* diag TS_P) {
	const uxx n = get_global_id(0);
	if(n>=(uxx)def_N||is_halo(n)) return;
)+"#ifdef SPARSE_TILES"+R(
	if(is_dead_tile(n, tile_slot)) return; // XL-Audit B1 (wie boden_eq)
)+"#endif"+R( // SPARSE_TILES
	const uint3 xyz = coordinates(n);
	if(nx==0u||xyz.x<1u||xyz.x>nx) return; // nur die Clamp-Schicht hinter dem Einlass; Test VOR dem flags-Load
	const uchar bo = flags[n]&TYPE_BO;
	if(bo==TYPE_S||bo==TYPE_E) return; // TYPE_MS wird BEHANDELT (s. o.)
	if(t%def_zaehl_takt==0ul) atomic_inc(&diag[21]); // Wirkpfad-Nachweis IM Binary (Iron Rule 3)
	uxx j[def_velocity_set]; neighbors(n, j);
	float fhn[def_velocity_set]; load_f(n, fhn, fi, j, t TS_A);
	float rho_local, ux, uy, uz; calculate_rho_u(fhn, &rho_local, &ux, &uy, &uz); // LOKALES rho (Druck erhalten); post-stream-load = Paare vertauscht -> u waere NEGIERT (nur rho ist invariant und wird genutzt, XL-B8)
)+"#if defined(KLEMM_BILANZ)&&defined(RHO_CLAMP)"+R(
	if(rho_local<=RHO_CLAMP_MIN||rho_local>=RHO_CLAMP_MAX) { // ★ 15.09.2026 Klemmen S0d (KLEMMEN-STUFE0-PLAN.md §1 Punkt 2): Faktor 1, weil f hier durch f_eq(rho_c) ERSETZT wird
		const float dq_ = fabs(rho_local-klemm_rho_roh(fhn));
		if(diag[266]<0xF0000000u) atomic_inc(&diag[266]);
		if((dq_>2.0f||(as_uint(dq_)&0x7F800000u)==0x7F800000u)&&diag[265]<0xF0000000u) atomic_inc(&diag[265]); // Kappung 2 (Host-Wickelschranke), Soll 0 -- ★ Audit 16.09.2026 (A-N6, H4): Exponent-Bittest, damit NaN/Inf gezaehlt wird
		atomic_add(&diag[rho_local<=RHO_CLAMP_MIN ? 267u : 268u], convert_uint_sat(fma(fmin(dq_, 2.0f), def_klemm_s, 0.5f)));
	}
)+"#endif"+R( // KLEMM_BILANZ
	float feq[def_velocity_set]; calculate_f_eq(rho_local, u_road, 0.0f, 0.0f, feq);
	store_f(n, feq, fi, j, t TS_A);
}
)
+R(kernel void update_fields)+"("+R(const global fpxx* fi, global rhoxx* rho, global velxx* u, const global uchar* flags, const ulong t, const float fx, const float fy, const float fz // ) { // calculate fields from DDFs
)+"#ifdef FORCE_FIELD"+R(
	, const global float* F, const global uint* f_maske // argument order is important (f_maske: F-Markerliste, 03.09.; im Vollfeld-Arm ungelesen)
)+"#endif"+R( // FORCE_FIELD
)+"#ifdef TEMPERATURE"+R(
	, const global fpxx* gi, global float* T // argument order is important
)+"#endif"+R( // TEMPERATURE
)+R( TS_P
)+") {"+R( // update_fields()
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute update_fields() on halo
)+"#ifdef SPARSE_TILES"+R(
	// FORK: Zellen in toten Tiles duerfen fi NIE anfassen. cell_base() gibt fuer sie defensiv 0
	// zurueck -- ohne diesen Ausstieg landen ihre Schreibzugriffe also in Slot 0 und ueberschreiben
	// die Daten einer echten aktiven Tile. Genau daran ist der erste T=8-Lauf divergiert (Cd 18.4).
	if(is_dead_tile(n, tile_slot)) return;
)+"#endif"+R(
	const uchar flagsn = flags[n];
	const uchar flagsn_bo=flagsn&TYPE_BO, flagsn_su=flagsn&TYPE_SU; // extract boundary and surface flags
	if(flagsn_bo==TYPE_S||flagsn_su==TYPE_G) return; // don't update fields for boundary or gas lattice points

	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	float fhn[def_velocity_set]; // local DDFs
	load_f(n, fhn, fi, j, t TS_A); // perform streaming (part 2)

)+"#ifdef MOVING_BOUNDARIES"+R(
	if(flagsn_bo==TYPE_MS) apply_moving_boundaries(fhn, j, u, flags); // apply Dirichlet velocity boundaries if necessary (reads velocities of only neighboring boundary cells, which do not change during simulation)
)+"#endif"+R( // MOVING_BOUNDARIES

	float rhon, uxn, uyn, uzn; // calculate local density and velocity for collision
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn); // calculate density and velocity fields from fi
	float fxn=fx, fyn=fy, fzn=fz; // force starts as constant volume force, can be modified before call of calculate_forcing_terms(...)

// ★ F-Null-Read-Gate: identisch zur stream_collide-Stelle gegated, damit F_NUR_SOLID genau
// EINE Bedeutung hat (kein F-Read an Nicht-Solid-Zellen) -- Build-Varianten-Konsistenz.
)+"#ifdef FORCE_FIELD"+R(
)+"#ifndef F_NUR_SOLID"+R(
	{ // separate block to avoid variable name conflicts
		const float3 Fn = load3_F(F, f_maske, n); // FORK: bbox-bewusst
		fxn += Fn.x; fyn += Fn.y; fzn += Fn.z;
	}
)+"#endif"+R( // F_NUR_SOLID
)+"#endif"+R( // FORCE_FIELD

)+"#ifdef TEMPERATURE"+R(
	{ // separate block to avoid variable name conflicts
		uxx j7[7]; // neighbors of D3Q7 subset
		neighbors_temperature(n, j7);
		float ghn[7]; // read from gA and stream to gh (D3Q7 subset, periodic boundary conditions)
		load_g(n, ghn, gi, j7, t); // perform streaming (part 2)
		float Tn;
		if(flagsn&TYPE_T) {
			Tn = T[n]; // apply preset temperature
		} else {
			Tn = 0.0f;
			for(uint i=0u; i<7u; i++) Tn += ghn[i]; // calculate temperature from g
			Tn += 1.0f; // add 1.0f last to avoid digit extinction effects when summing up gi (perturbation method / DDF-shifting)
			T[n] = Tn; // update temperature field
		}
		fxn -= fx*def_beta*(Tn-def_T_avg);
		fyn -= fy*def_beta*(Tn-def_T_avg);
		fzn -= fz*def_beta*(Tn-def_T_avg);
	}
)+"#endif"+R( // TEMPERATURE

	{ // separate block to avoid variable name conflicts
)+"#ifdef VOLUME_FORCE"+R( // apply force and collision operator, write to fi in video memory
		const float rho2 = 0.5f/rhon; // apply external volume force (Guo forcing, Krueger p.233f)
)+"#ifdef U_BETRAG"+R(
		{ const float uxb_ = fma(fxn, rho2, uxn), uyb_ = fma(fyn, rho2, uyn), uzb_ = fma(fzn, rho2, uzn); const float u2b_ = uxb_*uxb_+uyb_*uyb_+uzb_*uzb_; const float skb_ = u2b_>=def_u2max ? sqrt(def_u2max/u2b_) : 1.0f; uxn = uxb_*skb_; uyn = uyb_*skb_; uzn = uzb_*skb_; } // ★ Z2d: wie stream_collide (Kernel derzeit ohne Aufrufer)
)+"#else"+R( // U_BETRAG
		uxn = clamp(fma(fxn, rho2, uxn), -def_c, def_c); // limit velocity (for stability purposes)
		uyn = clamp(fma(fyn, rho2, uyn), -def_c, def_c); // force term: F*dt/(2*rho)
		uzn = clamp(fma(fzn, rho2, uzn), -def_c, def_c);
)+"#endif"+R( // U_BETRAG
)+"#else"+R( // VOLUME_FORCE
)+"#ifdef U_BETRAG"+R(
		{ const float u2b_ = uxn*uxn+uyn*uyn+uzn*uzn; const float skb_ = u2b_>=def_u2max ? sqrt(def_u2max/u2b_) : 1.0f; uxn *= skb_; uyn *= skb_; uzn *= skb_; } // ★ Z2d
)+"#else"+R( // U_BETRAG
		uxn = clamp(uxn, -def_c, def_c); // limit velocity (for stability purposes)
		uyn = clamp(uyn, -def_c, def_c); // force term: F*dt/(2*rho)
		uzn = clamp(uzn, -def_c, def_c);
)+"#endif"+R( // U_BETRAG
)+"#endif"+R( // VOLUME_FORCE
	}

)+"#ifdef EQUILIBRIUM_BOUNDARIES"+R(
	if(flagsn_bo!=TYPE_E) // only update fields for non-TYPE_E cells
)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
	{
		)+"#ifdef RHO_RAND"+R(
		{ const uxx rr_ = rr_idx(n); if(rr_<(uxx)def_RR_N) store_rho(rho, rr_, rhon); } // ★ C2b: toter Pfad, aber gebunden -- nie ausserhalb R1 schreiben
		)+"#else"+R(
		store_rho(rho, n, rhon); // update density field
		)+"#endif"+R( // RHO_RAND
		store3_u(u, n, (float3)(uxn, uyn, uzn)); // update velocity field
	}
} // update_fields()

// ★ FORK 2026-08-08: atomic_add_f stand bisher INNERHALB des FORCE_FIELD-Blocks und damit hinter
// dem Druck-Auslass. Der braucht es jetzt fuer die Flaechenmittel-Reduktion, also hierher gezogen --
// gleiche Definition, nur frueher und ohne die FORCE_FIELD-Bedingung.
)+R(void atomic_add_f(volatile global float* addr, const float val) {
)+"#if cl_nv_compute_capability>=20"+R( // use hardware-supported atomic addition on Nvidia GPUs with inline PTX assembly
	float ret;)+"asm volatile(\"atom.global.add.f32\t%0,[%1],%2;\":\"=f\"(ret):\"l\"(addr),\"f\"(val):\"memory\");"+R(
)+"#elif defined(__opencl_c_ext_fp32_global_atomic_add)"+R( // use hardware-supported atomic addition on some Intel GPUs
	atomic_fetch_add_explicit((volatile global atomic_float*)addr, val, memory_order_relaxed);
)+"#elif __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)"+R( // use hardware-supported atomic addition on some AMD GPUs
	__builtin_amdgcn_global_atomic_fadd_f32(addr, val);
)+"#else"+R( // fallback emulation: https://forums.developer.nvidia.com/t/atomicadd-float-float-atomicmul-float-float/14639/5
	float old = val; while((old=atomic_xchg(addr, atomic_xchg(addr, 0.0f)+old))!=0.0f);
)+"#endif"+R(
}
)+R(kernel void po_reduce_mean(const global rhoxx* rho, const global uint* po_interior, const uint N_po, global float* po_part) {
	// Mittelwert der Dichte ueber die INNENZELLEN der Auslassebene: erst im lokalen Speicher
	// zusammenfassen, dann schreibt jede Gruppe exklusiv ihren Slot. Die Endsumme bildet
	// po_final_mean in Indexordnung -- OHNE Atomik (Umbau 2026-08-24, siehe dort).
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	local float cache[cl_workgroup_size];
	// ★ Es wird rho-1 summiert, nicht rho. Ein Pruefer hat den Fehler der atomaren Serie
	// nachgerechnet: 4697 Beitraege von je 2,1e-4 auf einen Akkumulator, der bis 1,0 waechst, geben
	// 1,2e-5 Fehler im Mittelwert -- das sind 1,4e-3 in cp gegen die 0,15, um die es geht. Mit der
	// Abweichungsablage sind es 1e-9. Derselbe Kniff steht schon in calculate_rho_u
	// ("add 1.0f last to avoid digit extinction effects").
	// ★ 12.09.2026: load_drho, NICHT load_rho. Der Umweg ueber "+1, dann -1" wuerde den Wert auf das
	// float32-Raster bei 1,0 runden und damit genau den absoluten Boden von 5,96e-8 einbauen, gegen
	// den der Absatz darueber argumentiert. Ohne RHO_FP16 expandiert load_drho zu (rho[...]-1.0f) --
	// zeichengleich zu dem, was hier vorher stand.
)+"#ifdef RHO_RAND"+R(
	cache[lid] = gid<N_po ? load_drho(rho, rr_idx((uxx)po_interior[gid])) : 0.0f; // ★ C2b: po_interior liegt in R1 (C0-Waechter 1b)
)+"#else"+R(
	cache[lid] = gid<N_po ? load_drho(rho, po_interior[gid]) : 0.0f;
)+"#endif"+R( // RHO_RAND
	barrier(CLK_LOCAL_MEM_FENCE);
	for(uint s=1u; s<cl_workgroup_size; s*=2u) {
		if(lid%(2u*s)==0u) cache[lid] += cache[lid+s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// ★ 2026-08-24: OHNE ATOMIK. Vorher stand hier ein atomic_add_f je Arbeitsgruppe auf einen
	// EINZELNEN Float -- die Additionsreihenfolge haing damit an der Gruppenreihenfolge, und weil
	// po_mean jeden Schritt in die Randbedingung zurueckkoppelt, war das die einzige
	// loesungswirksame Ordnungsabhaengigkeit im Zeitschritt (gemessen: sechs identische
	// Kugellaeufe lieferten VIER verschiedene u-Feld-Hashes). Jetzt schreibt jede Gruppe
	// exklusiv ihren Slot, und po_final_mean summiert in Indexordnung. Bauart wortgleich zu
	// kraft_facetten_gpu (kernel.cpp:3365 ff.), das denselben Weg schon geht.
	// Kein !=0-Test noetig: jeder Slot wird jeden Schritt ueberschrieben.
	if(lid==0u) po_part[get_group_id(0)] = cache[0];
} // po_reduce_mean()

)+R(kernel void po_final_mean(const global float* po_part, const uint n_groups, const uint N_po, global float* po_mean) {
	// EIN Arbeitselement, Summe in FESTER Indexordnung -- damit ist das Ergebnis bitreproduzierbar.
	// Die Division steht am ENDE statt je Gruppe: 493 Additionen und EINE Division statt 494
	// Divisionen und 494 atomarer Additionen, also zugleich der numerisch bessere Weg (der
	// Fehler im Mittelwert faellt von rund 1,2e-5 auf die Groessenordnung 1e-10).
	if(get_global_id(0)!=0u) return;
	float s = 0.0f;
	for(uint g=0u; g<n_groups; g++) s += po_part[g];
	po_mean[0] = s/(float)N_po;
} // po_final_mean()

)+R(kernel void apply_pressure_outlet(global velxx* u, global rhoxx* rho, const global uint* po_cells, const global uint* po_interior, const uint N_po, const float rho_out, const float po_sigma, const global float* po_mean, const uint po_hart) {
	// FORK -- Druck-Auslass. Setzt an jeder Auslasszelle die vorgeschriebene Dichte und kopiert die
	// Geschwindigkeit aus der zugehoerigen Innenzelle (Nullgradient). Zusammen mit der TYPE_E-Logik in
	// stream_collide ergibt das f = f_eq(rho_out, u_innen): Dirichlet auf den Druck, Neumann auf u.
	//
	// Die Innenzelle kommt FERTIG vom Host, statt hier aus einer Richtung abgeleitet zu werden. Grund:
	// eine Zelle auf zwei Auslassflaechen (Kante) oder dreien (Ecke) hat keine eindeutige Richtung, und
	// ein Schritt entlang einer davon landet wieder auf dem Rand. Der Host loest das einmal und prueft
	// es nach; hier bleibt nur noch ein Kopiervorgang, an dem nichts schiefgehen kann.
	//
	// Jede Randzelle kommt genau einmal in der Liste vor -- vom Host geprueft. Ohne das schrieben zwei
	// Work-Items dieselbe Zelle, und welches zuletzt gewinnt, ist undefiniert.
	// ★★ WEICHER DRUCKANKER, 2026-08-08. Vorher stand hier rho[n] = rho_out, also eine harte Klemme.
	// Ein Pruefer hat die Randregel in D1Q3 exakt nachgebaut und den Reflexionsgrad gemessen:
	//   rho UND u fest (alter Einlass) ....... R = 0,27..0,30   -- reiner Loeschoperator
	//   rho fest, u extrapoliert (DIESER Rand) R = 0,98..0,99   -- fast perfekter Spiegel
	//   rho extrapoliert, u fest ............. R = 1,00         -- vollkommener Spiegel
	// Sobald EINE Groesse aus der Innenzelle zurueckgelesen wird, entsteht eine Rueckkopplung:
	// Stoerung -> u[m] -> f_eq -> Nachbarzelle. Aus dem Absorber wird ein Spiegel.
	// Der weiche Anker relaxiert stattdessen mit der Rate sigma gegen rho_out:
	//   sigma  1,0    0,5    0,2    0,1    0,05   0,02   0,01
	//   R      0,981  0,919  0,726  0,526  0,339  0,167  0,092
	// Der Anker bleibt erhalten: bei sigma = 0,02 ist die Zeitkonstante 50 Schritte = 2,0 ms gegen
	// eine Durchspuelung von 0,409 s -- also 200-mal schneller als die Stroemung.
	// ★ KORRIGIERT (Pruefer-Befund): hier stand "sigma = 1 ist bit-genau der bisherige Zustand".
	// Mit dem Flaechenmittel-Anker stimmt das nicht mehr -- der Kontrollarm ist jetzt po_hart = 1,
	// siehe unten. Und die R-Tabelle wurde in D1Q3 gemessen, wo die Auslassebene EINE Zelle hat und
	// der Flaechenmittel-Anker in den Zell-Anker entartet; fuer den neuen Rand gilt sie nur fuer die
	// ebenen-gleichfoermige Mode.
	const uint gid = get_global_id(0);
	if(gid>=N_po) return;
	const ulong n = (ulong)po_cells[gid], m = (ulong)po_interior[gid]; // ★ 08.09. Listen sind uint (VRAM); der Index bleibt ulong
	// ★★ ANKER AUF DEN FLAECHENMITTELWERT, nicht auf jede Zelle. 2026-08-08.
	// Vorher wurde rho JEDER Auslasszelle einzeln gegen rho_out gezogen. Beim Fahrzeug sitzt der feine
	// Auslass 0,449 Fahrzeuglaengen hinter dem Heck, also im Totwasser -- dort erzwang rho_out = 1 auf
	// 300 564 Zellen ein cp = 0, wo etwa -0,15 hingehoert. Der Basisdruck ist beim Fahrzeug der
	// groesste Einzelbeitrag zu Cd; das ist ein systematischer Fehler, unabhaengig von jeder Akustik.
	// Jetzt wird nur noch der MITTELWERT der Ebene verankert: rho_neu = rho_innen + sigma*(rho_out -
	// <rho_innen>). Der Druck ist damit global festgelegt, seine VERTEILUNG ueber die Ebene aber frei.
	// po_mean[0] traegt den ABWEICHUNGS-Mittelwert <rho-1>, nicht <rho> -- siehe po_reduce_mean.
	//
	// ★ KONTROLLARM, 2026-08-08 (Pruefer-Befund, und es war mein Fehler): mit dem Flaechenmittel ist
	// sigma = 1 NICHT mehr der alte Zustand. Alt war rho[n] = (1-sigma)*rho[m] + sigma*rho_out, bei
	// sigma = 1 also die harte Klemme. Neu ist es Nullgradient plus Versatz -- eine ANDERE Bedingung,
	// und kein sigma-Wert bringt die alte zurueck. Der Kontrollarm war damit still verschwunden, und
	// der Default lief im neuen Regime, ohne dass es jemandem aufgefallen waere.
	// po_hart = 1 stellt den alten Rand bit-genau wieder her (CFD_PO_HART=1).
	// ★ 12.09.2026: BEWUSST load_rho und nicht load_drho. In Abweichungsraeumen zu rechnen waere hier
	// genauer, wuerde aber die Arithmetik des Arms OHNE RHO_FP16 aendern -- und dessen Bitgleichheit
	// zum Stand davor ist das einzige Sicherheitsnetz dieses Umbaus.
	// BERICHTIGT 12.09. (Pruefagent, NIEDRIG): hier stand, der Boden von 5,96e-8 aus der Addition
	// von 1 liege "unter der Quantisierung selbst". Das gilt erst ab |rho-1| > 1,2e-4 (Gleichstand
	// bei 5,96e-8 / 2^-11). Am Druckauslass wird rho konstruktiv gegen rho_out gezogen, also gerade
	// in die Zone KLEINER |rho-1| -- dort ist der Additionsboden fuehrend. Die Entscheidung bleibt
	// trotzdem richtig: 6e-8 in rho sind mit dem Umrechenfaktor rho->cp von rund 117 etwa 7e-6 in
	// cp, gegen die 0,15, um die es geht.
)+"#ifdef RHO_RAND"+R(
	{ const uxx rn_ = rr_idx((uxx)n), rm_ = rr_idx((uxx)m); // ★ C2b: Randzelle und Innenzelle liegen in R1 (C0-Waechter 1a/1b)
	  store_rho(rho, rn_, po_hart!=0u ? fma(po_sigma, rho_out-load_rho(rho, rm_), load_rho(rho, rm_))
	                                  : fma(po_sigma, (rho_out-1.0f)-po_mean[0], load_rho(rho, rm_))); }
)+"#else"+R(
	store_rho(rho, n, po_hart!=0u ? fma(po_sigma, rho_out-load_rho(rho, m), load_rho(rho, m))
	                              : fma(po_sigma, (rho_out-1.0f)-po_mean[0], load_rho(rho, m)));
)+"#endif"+R( // RHO_RAND
	store_u(u, n, load_u(u, m));
	store_u(u, def_N+(ulong)n, load_u(u, def_N+(ulong)m));
	store_u(u, 2ul*def_N+(ulong)n, load_u(u, 2ul*def_N+(ulong)m));
} // apply_pressure_outlet()

)+R(kernel void apply_velocity_inlet(global rhoxx* rho, const global ulong* vi_cells, const global ulong* vi_interior, const uint N_vi) {
	// FORK -- Spiegelbild des Druck-Auslasses. Dort wird rho vorgeschrieben und u aus der Innenzelle
	// genommen; hier wird u vorgeschrieben (das erledigt der TYPE_E-Zweig in stream_collide) und rho
	// aus der Innenzelle uebernommen. Damit ist je Rand genau EINE Groesse vorgegeben.
	// Warum das noetig ist, gemessen am leeren groben Kanal: mit vorgeschriebenem rho UND u ist der
	// Rand ueberbestimmt, reflektiert jede ankommende Druckwelle vollstaendig und schiebt die
	// Massendifferenz jeden Schritt in die erste Fluidzelle dahinter. Genau dort sass die Stoerung.
	const uint gid = get_global_id(0);
	if(gid>=N_vi) return;
	const ulong n = vi_cells[gid], m = vi_interior[gid];
	// Reines Kopieren. Dass das unter RHO_FP16 NICHT driftet, haengt am Fixpunkt der Wandlung
	// (Beweis an rho_pack in lbm.hpp): store_rho(load_rho(w)) liefert dasselbe Wort zurueck.
)+"#ifdef RHO_RAND"+R(
	store_rho(rho, rr_idx((uxx)n), load_rho(rho, rr_idx((uxx)m))); // ★ C2b: nur Fernfeld nutzt VI, dort gibt es kein RHO_RAND; Nahfeld-VI sperrt C0-Waechter 1c
)+"#else"+R(
	store_rho(rho, n, load_rho(rho, m));
)+"#endif"+R( // RHO_RAND
} // apply_velocity_inlet()

// =====================================================================================
// FORK -- Doppel-Domaene: Kopplung grobes Fernfeld -> feines Nahfeld.
//
// ZWECK. Der Fahrzeugfall braucht zweierlei, das sich in EINEM Gitter widerspricht: feine
// Aufloesung am Fahrzeug und einen Querschnitt, der gross genug ist, dass die Versperrung
// unter der Windkanalpraxis von 5 % bleibt. Mit einem Gitter kostet die zweite Forderung
// die erste. Zwei Gitter loesen das: ein grobes, weites Fernfeld traegt die Aussenstroemung,
// ein feines Nahfeld um das Fahrzeug bekommt seine Raender aus dem Fernfeld vorgeschrieben.
//
// VERFAHREN. Pro Fernfeld-Zeitschritt (dt_c = ratio*dt_f):
//   1. extract_plane_macros liest (rho, u) auf den fuenf Fernfeld-Ebenen, die genau auf den
//      Aussenflaechen der Nahfeld-Box liegen (der Boden z=0 ist beiden gemeinsam, kein Rand).
//   2. drive_boundary_cubic_lift interpoliert diese Ebene kubisch auf Nahfeld-Aufloesung und
//      schreibt sie in die TYPE_E-Randzellen des Nahfelds.
//   3. Das Nahfeld rechnet `ratio` Schritte, dann von vorn.
// Die Interpolation ist die 4-Punkt-Lagrange-Formel aus Lagrava/Latt/Chopard, JCP 231 (2012),
// Gl. 38, mit den einseitigen 3-Punkt-Randformeln aus Gl. 39. Deckungspunkt-Konvention:
// fine_extent = (coarse_extent - 1) * ratio + 1, d. h. jeder grobe Punkt faellt exakt auf einen
// feinen -- fuer s = 0 ist die Interpolation die Identitaet.
//
// WAS BEWUSST NICHT DRIN IST -- und warum. Das volle Lagrava-Latt-Verfahren gibt zusaetzlich
// den Nichtgleichgewichtsanteil f_neq mit einem Faktor omega_c/(r*omega_f) weiter und
// dezimiert das feine Feld zurueck ins grobe. Beides ist hier absichtlich nicht portiert:
//   * f_neq waere WIRKUNGSLOS. Die Nahfeld-Randzellen sind TYPE_E; stream_collide setzt dort
//     f = f_eq(rho, u) (siehe Zeile mit "flagsn_bo==TYPE_E ? feq[i]"). Ein eingespritztes f_neq
//     wuerde im selben Schritt ueberschrieben. Der alte Baum hat f_neq extrahiert, geliftet und
//     skaliert -- und dann verworfen; gemessen an der Wirkung war das toter Code.
//   * Die Rueckkopplung fein -> grob war im alten Baum am 2026-06-14 mit Begruendung abgeschaltet:
//     das grobe Gitter (4x groeber) kann die aufgeloeste Nachlaufstroemung nicht aufnehmen, die
//     dezimierten Daten wirkten als Barriere, um die das Fernfeld herumstroemte.
// Die Kopplung ist damit EINWEG: das Fernfeld diktiert dem Nahfeld die Raender, nicht umgekehrt.
// Das ist eine Naeherung, und sie ist es wert, benannt zu werden: die Versperrung des Fahrzeugs
// wirkt nur ueber das grobe Gitter zurueck, in dem das Fahrzeug ebenfalls voxelisiert ist.
// =====================================================================================

)+R(void cubic_lift_weights(const uint f, const uint ratio, const uint c_ext, int* idx, float* w) {
	// 4-Punkt-Lagrange-Gewichte fuer die Position f auf dem feinen Gitter (Lagrava Gl. 38).
	// c_ext = Zahl der groben Stuetzstellen auf dieser Achse. Am Rand, wo vier Stuetzstellen nicht
	// verfuegbar sind, fallen wir auf die einseitige 3-Punkt-Formel (Gl. 39) bzw. auf linear zurueck.
	// Alle vier Zweige sind Lagrange-Basen, summieren sich also exakt zu 1 -- ein konstantes Feld
	// bleibt konstant, was der Test in der Verifikation ausnutzt.
	const int ic = (int)(f/ratio); // Index der groben Stuetzstelle links davon
	const int s  = (int)(f%ratio); // Unterteilung dazwischen; s==0 heisst: feiner Punkt IST grober Punkt
	if(s==0) { idx[0]=idx[1]=idx[2]=idx[3]=ic; w[0]=1.0f; w[1]=w[2]=w[3]=0.0f; }
	else {
		const float tt = (float)s/(float)ratio;
		const bool L = (ic==0), Rr = (ic+2>=(int)c_ext); // links/rechts fehlt eine Stuetzstelle
		if(L&&!Rr)      { idx[0]=ic;   idx[1]=ic+1; idx[2]=ic+2; idx[3]=ic+2; w[0]=(tt-1.0f)*(tt-2.0f)*0.5f;       w[1]=-tt*(tt-2.0f);                       w[2]=tt*(tt-1.0f)*0.5f;             w[3]=0.0f; }
		else if(Rr&&!L) { idx[0]=ic-1; idx[1]=ic;   idx[2]=ic+1; idx[3]=ic+1; w[0]=tt*(tt-1.0f)*0.5f;              w[1]=(tt+1.0f)*(1.0f-tt);                 w[2]=(tt+1.0f)*tt*0.5f;             w[3]=0.0f; }
		else if(L&&Rr)  { idx[0]=ic;   idx[1]=ic+1; idx[2]=ic+1; idx[3]=ic+1; w[0]=1.0f-tt;                        w[1]=tt;                                  w[2]=0.0f;                          w[3]=0.0f; }
		else            { idx[0]=ic-1; idx[1]=ic;   idx[2]=ic+1; idx[3]=ic+2; w[0]=-tt*(tt-1.0f)*(tt-2.0f)/6.0f;   w[1]=(tt+1.0f)*(tt-1.0f)*(tt-2.0f)*0.5f;  w[2]=-(tt+1.0f)*tt*(tt-2.0f)*0.5f;  w[3]=(tt+1.0f)*tt*(tt-1.0f)/6.0f; }
	}
	for(uint i=0u; i<4u; i++) { if(idx[i]<0) idx[i]=0; if(idx[i]>=(int)c_ext) idx[i]=(int)c_ext-1; }
} // cubic_lift_weights()

)+R(uxx plane_cell_index(const uint gid, const uint plane_axis, const uint origin_x, const uint origin_y, const uint origin_z, const uint extent_a, const uint extent_b) {
	// Bildet den flachen Ebenen-Index auf den Zellindex der Domaene ab. Gemeinsam benutzt, damit
	// Auslesen und Einspeisen garantiert dieselbe Zuordnung verwenden -- ein Versatz zwischen beiden
	// waere ein Fehler, den keine Norm und kein Kraftverlauf sichtbar machen wuerde.
	const uint a_idx = gid % extent_a;
	const uint b_idx = gid / extent_a;
	uint x, y, z;
	if(plane_axis==0u)      { x = origin_x;         y = origin_y+a_idx; z = origin_z+b_idx; } // X-normal: a=y, b=z
	else if(plane_axis==1u) { x = origin_x+a_idx;   y = origin_y;       z = origin_z+b_idx; } // Y-normal: a=x, b=z
	else                    { x = origin_x+a_idx;   y = origin_y+b_idx; z = origin_z;       } // Z-normal: a=x, b=y
	return (uxx)x + (uxx)y*(uxx)def_Nx + (uxx)z*(uxx)def_Nx*(uxx)def_Ny;
} // plane_cell_index()

)+R(kernel void extract_plane_macros(const global rhoxx* rho, const global velxx* u, global float* out,
	const uint plane_axis, const uint origin_x, const uint origin_y, const uint origin_z,
	const uint extent_a, const uint extent_b) {
	// Liest (rho, u_x, u_y, u_z) auf einer achsen-normalen Ebene in einen dichten Puffer.
	// HINWEIS (Kernel-Audit 2026-08-22, NIEDRIG): der N2F-Waechter-Extract liest das u-FELD,
	// das stream_collide VOR dem Blend desselben Schritts schrieb -- die Waechterzahlen sind
	// der PRAE-Blend-Stand und hinken fi um einen Blend nach. Reine Diagnose, kein Wirkpfad.
	// Liest die Makro-Felder DIREKT statt sie aus den DDFs zu rekonstruieren -- das setzt UPDATE_FIELDS
	// voraus, weil rho/u sonst auf dem Stand der letzten update_fields()-Anforderung stehen.
	// Der Aufrufer prueft das; hier waere die Pruefung nicht ausdrueckbar.
	const uint gid = get_global_id(0);
	if(gid>=extent_a*extent_b) return;
	const ulong o = (ulong)gid*4ul;
	const uxx n = plane_cell_index(gid, plane_axis, origin_x, origin_y, origin_z, extent_a, extent_b);
	if(n>=(uxx)def_N) { out[o]=1.0f; out[o+1ul]=0.0f; out[o+2ul]=0.0f; out[o+3ul]=0.0f; return; }
)+"#ifdef RHO_RAND"+R(
	out[o+0ul] = as_float(0x7FC00000u); // ★ C2b/C2c: Spalte 0 ist unter RHO_RAND INSGESAMT ungueltig (auch R1-Fluidzellen x<Nx-2 tragen nur die Saat) -- NaN-Wort (Hausmuster schale_extract; ein NAN-Makro darf unter -cl-finite-math-only weggefaltet werden), der Host nimmt rho aus rho_ausgabe_ebene
)+"#else"+R(
	out[o+0ul] = load_rho(rho, n);
)+"#endif"+R( // RHO_RAND
	out[o+1ul] = load_u(u, n);
	out[o+2ul] = load_u(u, def_N+(ulong)n);
	out[o+3ul] = load_u(u, 2ul*def_N+(ulong)n);
} // extract_plane_macros()

)+R(kernel void rho_rek_ebene)+"("+R(const global fpxx* fi, const global rhoxx* rho, const global velxx* u, const global uchar* flags, const ulong t, global float* out, global rhoxx* out_wort,
	const uint plane_axis, const uint origin_x, const uint origin_y, const uint origin_z, const uint extent_a, const uint extent_b, const uint modus, global uint* rho_clamp_hits
)+R( TS_P
)+") {"+R( // rho_rek_ebene()
	// ★ 15.09.2026 C2a: modus 0 = wie C1 (Identitaetspruefung, MIT apply_moving_boundaries, out[1] = rho roh).
	// modus 1 = Nachkollisionsmodus (Aufruf mit t-1): OHNE apply_moving_boundaries -- die MS-Korrektur steckt schon in der
	// Summe der Nachkollisions-Populationen (die Kollision erhaelt die Masse) -- und out[1] = Summe |f~_i| fuer die
	// FP16S-Rundungsschranke der Pruefung (RHO_RAND-C2-PLAN.md §1.2, §4.4).
	// ★ 15.09.2026 RHO_RAND C1 (RHO_RAND-PLAN.md §5/§11): rho einer achsen-normalen Ebene AUS DEN DDFs, ohne
	// den rho-Puffer zu lesen -- ausser dort, wo stream_collide selbst ihn liest (TYPE_E) oder nie schreibt (TYPE_S).
	// Aufgerufen mit dem AKTUELLEN Domaenen-t (nach increment_time_step): load_f liest dann genau die Slots, die
	// stream_collide(t) gleich lesen wird, das Ergebnis ist also das rho, das stream_collide(t) speichern wird --
	// bitgleich ueberall dort, wo zwischen load_f und calculate_rho_u nichts an fhn aendert. Wandmodelle, die fhn VOR
	// calculate_rho_u umschreiben, weichen nur ab, wenn sie die MASSE nicht erhalten (Pruefpass C1 zu Plan K4):
	// iMEM mit FAC_ALPHA>=1 erhaelt sie analytisch, FAC_ALPHA=0 und die ELIBB-Blende bei q != 0,5 nicht.
	// TYPE_E: hier wird der Puffer GELESEN. Am Druckauslass schreibt apply_pressure_outlet ihn zu BEGINN des naechsten
	// Schritts neu (do_time_step: Einlass, Auslass, dann stream_collide) -- dort hinkt dieser Wert einen Schritt nach.
	// Ausgabe je Ebenenzelle (4 floats): [0] rho rekonstruiert (MIT RHO_CLAMP, wie gespeichert), [1] rho roh
	// (Summe ohne Klemme), [2] Klasse (0 Fluid, 1 TYPE_E, 2 TYPE_S, 3 TYPE_MS, 4 tote Kachel, 5 ausserhalb),
	// [3] heutiger Pufferwert load_rho. out_wort traegt [0] als GERAETE-gepacktes Speicherwort (vstore_half_rte),
	// damit der Host Woerter vergleichen kann -- der Hostpacker rundet anders (lbm.hpp, rho_unpack).
	// Die Klemmzaehler 0/1 stehen in stream_collide AUSSERHALB von calculate_rho_u und werden hier bewusst
	// NICHT mitgezaehlt. Zaehler: [217] Besuche (UNGEGATET, Ist=Soll = Ebenenzellen je Aufruf), [218] davon TYPE_E.
	const uint gid = get_global_id(0);
	if(gid>=extent_a*extent_b) return;
	const ulong o = (ulong)gid*4ul;
	if(rho_clamp_hits[217]<0xF0000000u) atomic_inc(&rho_clamp_hits[217]);
	const uxx n = plane_cell_index(gid, plane_axis, origin_x, origin_y, origin_z, extent_a, extent_b);
	if(n>=(uxx)def_N) { out[o]=1.0f; out[o+1ul]=1.0f; out[o+2ul]=5.0f; out[o+3ul]=1.0f; store_rho(out_wort, gid, 1.0f); return; }
	const uchar flagsn_bo = flags[n]&TYPE_BO;
	float rhon = 1.0f, rho_roh = 1.0f, klasse = 0.0f;
	bool rekonstruieren = true;
)+"#ifdef SPARSE_TILES"+R(
	if(is_dead_tile(n, tile_slot)) { klasse = 4.0f; rekonstruieren = false; }
)+"#endif"+R( // SPARSE_TILES
	if(rekonstruieren&&flagsn_bo==TYPE_S) { klasse = 2.0f; rekonstruieren = false; }
	)+"#ifdef EQUILIBRIUM_BOUNDARIES"+R(
	if(rekonstruieren&&flagsn_bo==TYPE_E) {
)+"#ifdef RHO_RAND"+R(
		rhon = load_rho(rho, rr_idx(n)); rho_roh = rhon; klasse = 1.0f; rekonstruieren = false; // ★ C2b (Pruefkernel; unter RHO_RAND sperrt der Host ihn)
)+"#else"+R(
		rhon = load_rho(rho, n); rho_roh = rhon; klasse = 1.0f; rekonstruieren = false;
)+"#endif"+R( // RHO_RAND
		if(rho_clamp_hits[218]<0xF0000000u) atomic_inc(&rho_clamp_hits[218]);
	}
	)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES -- ohne sie rekonstruiert stream_collide TYPE_E-Zellen wie Fluid (Pruefpass C1, N5)
	if(rekonstruieren) {
		uxx j[def_velocity_set];
		neighbors(n, j);
		float fhn[def_velocity_set];
		load_f(n, fhn, fi, j, t TS_A);
)+"#ifdef MOVING_BOUNDARIES"+R(
		if(flagsn_bo==TYPE_MS) { if(modus==0u) apply_moving_boundaries(fhn, j, u, flags); klasse = 3.0f; }
)+"#endif"+R( // MOVING_BOUNDARIES
		if(modus==0u) {
			rho_roh = fhn[0];
			for(uint i=1u; i<def_velocity_set; i++) rho_roh += fhn[i];
			rho_roh += 1.0f; // dieselbe Summenreihenfolge wie calculate_rho_u, nur ohne RHO_CLAMP
		} else {
			rho_roh = fabs(fhn[0]);
			for(uint i=1u; i<def_velocity_set; i++) rho_roh += fabs(fhn[i]); // modus 1: Summe |f~_i| (Schranke), NICHT rho
		}
		float uxn, uyn, uzn;
		calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);
	}
	out[o+0ul] = rhon;
	out[o+1ul] = rho_roh;
	out[o+2ul] = klasse;
)+"#ifdef RHO_RAND"+R(
	out[o+3ul] = as_float(0x7FC00000u); // ★ C2c: Pruefkernel unter RHO_RAND host-seitig gesperrt; heutiger Pufferwert existiert nicht mehr
)+"#else"+R(
	out[o+3ul] = load_rho(rho, n);
)+"#endif"+R( // RHO_RAND
	store_rho(out_wort, gid, rhon);
} // rho_rek_ebene()

)+R(kernel void rho_ausgabe_ebene)+"("+R(const global fpxx* fi, const global rhoxx* rho, const global uchar* flags, const ulong t, global float* out,
	const uint plane_axis, const uint origin_x, const uint origin_y, const uint origin_z, const uint extent_a, const uint extent_b, const uint zaehlen, global uint* rho_clamp_hits
)+R( TS_P
)+") {"+R( // rho_ausgabe_ebene()
	// ★ 15.09.2026 RHO_RAND C2a (RHO_RAND-C2-PLAN.md §2.5, Entscheidung (b) Heiko): rho einer Ebene fuer die AUSGABE (Slices,
	// Sonde, VTK) als Summe der eigenen Nachkollisions-Populationen. Der Host uebergibt t = get_t()-1: load_f liest dann die
	// Populationen, die stream_collide(t) nach der Kollision abgelegt hat (Paare getauscht, Summe gleich). KEIN
	// apply_moving_boundaries (die Korrektur steckt schon in der Summe), calculate_rho_u MIT RHO_CLAMP wie das heutige
	// Speicherwort; die Klemmzaehler 0/1 werden bewusst NICHT mitgezaehlt. TYPE_E liest den Puffer (heutiger Stand),
	// TYPE_S = 1 (dort schreibt niemand). Zaehler nur mit zaehlen != 0: [219] Besuche, [220] davon TYPE_E -- nicht je VTK-Zelle.
	const uint gid = get_global_id(0);
	if(gid>=extent_a*extent_b) return;
	if(zaehlen!=0u&&rho_clamp_hits[219]<0xF0000000u) atomic_inc(&rho_clamp_hits[219]);
	const uxx n = plane_cell_index(gid, plane_axis, origin_x, origin_y, origin_z, extent_a, extent_b);
	if(n>=(uxx)def_N) { out[gid] = 1.0f; return; }
)+"#ifdef SPARSE_TILES"+R(
	if(is_dead_tile(n, tile_slot)) { out[gid] = 1.0f; return; }
)+"#endif"+R( // SPARSE_TILES
	const uchar flagsn_bo = flags[n]&TYPE_BO;
	if(flagsn_bo==TYPE_S) { out[gid] = 1.0f; return; }
	)+"#ifdef EQUILIBRIUM_BOUNDARIES"+R(
	if(flagsn_bo==TYPE_E) {
)+"#ifdef RHO_RAND"+R(
		{ const uxx rr_ = rr_idx(n);
		  if(rr_>=(uxx)def_RR_N&&rho_clamp_hits[216]<0xF0000000u) atomic_inc(&rho_clamp_hits[216]); // Soll 0
		  out[gid] = load_rho(rho, rr_); }
)+"#else"+R(
		out[gid] = load_rho(rho, n);
)+"#endif"+R( // RHO_RAND
		if(zaehlen!=0u&&rho_clamp_hits[220]<0xF0000000u) atomic_inc(&rho_clamp_hits[220]);
		return;
	}
	)+"#endif"+R( // EQUILIBRIUM_BOUNDARIES
	uxx j[def_velocity_set];
	neighbors(n, j);
	float fhn[def_velocity_set];
	load_f(n, fhn, fi, j, t TS_A);
	float rhon, uxn, uyn, uzn;
	calculate_rho_u(fhn, &rhon, &uxn, &uyn, &uzn);
	out[gid] = rhon;
} // rho_ausgabe_ebene()

)+R(kernel void extract_plane_flags(const global uchar* flags, global uchar* out,
	const uint plane_axis, const uint origin_x, const uint origin_y, const uint origin_z,
	const uint extent_a, const uint extent_b) {
	// Zwilling von extract_plane_macros fuer das flags-Feld (Slice-Ebenen-Read 2026-08-26).
	// Die CSV-Spalten (nah_solid, Sonden-Solid-Bit) haengen an den DEVICE-flags inklusive
	// TYPE_MS-Saum aus dem initialize-Kernel (Pruefagent NIEDRIG-1: update_moving_boundaries
	// laeuft im Fahrzeugfall nie) -- der Host-Voxelstand allein waere NICHT wertgleich.
	const uint gid = get_global_id(0);
	if(gid>=extent_a*extent_b) return;
	const uxx n = plane_cell_index(gid, plane_axis, origin_x, origin_y, origin_z, extent_a, extent_b);
	out[gid] = (n>=(uxx)def_N) ? (uchar)TYPE_S : flags[n];
} // extract_plane_flags()

)+R(kernel void drive_boundary_cubic_lift(global rhoxx* rho, global velxx* u, const global uchar* flags,
	const global float* coarse_plane,
	const uint plane_axis, const uint origin_x, const uint origin_y, const uint origin_z,
	const uint extent_a, const uint extent_b,
	const uint coarse_a, const uint coarse_b, const uint ratio, global uint* hits) {
	// Kubische Interpolation der groben Ebene auf die feine Randebene, direkt in rho[]/u[] geschrieben.
	// Lift und Einspeisung sind bewusst EIN Kernel: ein Zwischenpuffer in feiner Aufloesung waere
	// mehrere hundert MB gross und wuerde nur weitergereicht.
	const uint gid = get_global_id(0);
	if(gid>=extent_a*extent_b) return;
	const uxx n = plane_cell_index(gid, plane_axis, origin_x, origin_y, origin_z, extent_a, extent_b);
	if(n>=(uxx)def_N) return;
	if((flags[n]&TYPE_BO)!=TYPE_E) return; // nur reine Gleichgewichts-Randzellen; Boden und Fahrzeug bleiben unberuehrt
	int ia[4], ib[4]; float wa[4], wb[4];
	cubic_lift_weights(gid % extent_a, ratio, coarse_a, ia, wa);
	cubic_lift_weights(gid / extent_a, ratio, coarse_b, ib, wb);
	float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
	for(uint jj=0u; jj<4u; jj++) for(uint ii=0u; ii<4u; ii++) {
		const float wij = wa[ii]*wb[jj];
		if(wij==0.0f) continue;
		const ulong cb = ((ulong)ib[jj]*(ulong)coarse_a + (ulong)ia[ii])*4ul;
		v[0] += wij*coarse_plane[cb+0ul];
		v[1] += wij*coarse_plane[cb+1ul];
		v[2] += wij*coarse_plane[cb+2ul];
		v[3] += wij*coarse_plane[cb+3ul];
	}
)+"#ifdef KLEMM_BILANZ"+R(
	// ★ 15.09.2026 Klemmen Z2b (KLEMMEN-STUFE2-PLAN.md §2.3): Slot 300 = Lift-rho ausserhalb der BILDHUELLE 1 +- Lambda^2*(RHO_CLAMP_MAX-1) =
	// (0,21875; 1,78125). Lambda = 1,25 je Achse (Summe |w| der Lagrange-4-Punkt-/3-Punkt-Gewichte = 1 + t(1-t), Pruefbefund Z2b N2), Huelle GESCHLOSSEN (Randwerte legal, N1), Eingang geklemmt -> Soll 0; das Tor darunter verwirft heute
	// auch legale Werte in (0,219; 0,5].
	if((v[0]<def_tor_lo||v[0]>def_tor_hi)&&hits[300]<0xF0000000u) atomic_inc(&hits[300]);
	// ★ Audit-Schleife 16.09.2026, Befund B1 (HOCH): KONSTANTENSPIEGEL fuer das Tor. CFD_TOR_HUELLE verengte Tor und Waechter,
	// ohne dass ein einziger Zaehler belegte, dass die Verengung UEBERSETZT wurde -- sein einziges Soll war eine Null ([270] = 0),
	// und der einzige Positivtest (Haken 5) ist mit TOR_HUELLE per print_error verboten. Ein Schalter ohne feuernden Wirkpfad ist
	// in diesem Projekt ein harter Fehler. Hier schreibt der Kernel die uebersetzten Torgrenzen als Festkomma zurueck, der Host
	// vergleicht sie mit seiner eigenen Rechnung (Ist=Soll statt Nullbeweis; Soll [215] > 0 => [301] > 0).
	// KEIN atomic: alle Threads schreiben denselben konstanten Wert, das Rennen ist wirkungslos. Gelesen wird auf dem HOST --
	// der Store kann also nicht wegoptimiert werden (Lehre "Rueckleser im Kernel misst nichts", 12.09.2026).
	hits[301] = (uint)fma(def_tor_gate_lo, def_klemm_s, 0.5f);
	hits[302] = (uint)fma(def_tor_gate_hi, def_klemm_s, 0.5f);
)+"#endif"+R( // KLEMM_BILANZ
	// Unplausibles NICHT durchreichen: lieber den vorigen Randwert stehen lassen als das Nahfeld vergiften.
	// ★ Z2e/Z2f: Torgrenzen emittiert -- Vorgabe (0,5; 2,0) wie bisher, CFD_TOR_HUELLE=1 die Bildhuelle (Tor wird Invariante, Soll [270] = 0),
	// CFD_RHO_HUELLE=1 die numerische Huelle (dort gibt es kein Bild, das Tor darf greifen, Plan §2.3).
	if(!(v[0]>def_tor_gate_lo&&v[0]<def_tor_gate_hi)) {
)+"#ifdef KLEMM_BILANZ"+R(
		if(hits[270]<0xF0000000u) atomic_inc(&hits[270]); // ★ 15.09.2026 Klemmen S0d: Lift-rho-Tor griff, der vorige Randwert bleibt stehen
)+"#endif"+R( // KLEMM_BILANZ
		return;
	}
	// ★ Gross-Audit M: isfinite ist unter -cl-finite-math-only toter Code (Compiler faltet zu true) --
	// Bit-Test auf Exponent 0xFF faengt NaN/Inf treiberunabhaengig.
	if((as_uint(v[0])&0x7F800000u)==0x7F800000u||(as_uint(v[1])&0x7F800000u)==0x7F800000u||(as_uint(v[2])&0x7F800000u)==0x7F800000u||(as_uint(v[3])&0x7F800000u)==0x7F800000u) return; // R2: auch rho bit-testen (Bereichsvergleich ist unter finite-math NaN-unzuverlaessig)
	// ★ TODO 2 Schritt 4 (12.09.2026): Betragstor auf u, Zwilling zum rho-Tor zwei Zeilen darueber.
	// stream_collide und update_fields klemmen beide auf +-def_c, bevor sie speichern; dieser hier
	// nicht. BERICHTIGT 12.09. abends (Pruefer C): er ist nicht der EINZIGE ungeklemmte -- das sind
	// auch voxelize_mesh (schreibt ein hostgegebenes u_set) und insert_rho_u_flags (uebernimmt
	// Halowerte einer Nachbardomaene). Er ist der einzige, der einen WERT AUSRECHNET statt einen
	// durchzureichen, und deshalb der einzige, der ueberschwingen kann. Die kubische Interpolation
	// kann ueberschwingen; der Eingang ist zwar selbst geklemmt, aber die Lagrange-Interpolation traegt Gewichte
	// ausserhalb [0,1]. Unter U_FP16 wuerde ein Wert ab |u| = 1,99902 still nach +-inf saettigen.
	// Schwelle 1,0 wie am Huellenwaechter Slot 212, Wirkpfad-Zaehler Slot 214, Soll 0 -- und weil er
	// 0 ist, bleibt der FP32-Arm bitgleich: das Tor greift nie, es beweist nur, dass es nie greift.
	// ★ BERICHTIGT 12.09. abends (Pruefer B, HOCH): dieses Tor kann konstruktiv NIE greifen, und das
	// gehoert hierhin statt "Soll 0". Der Eingang ist auf +-def_c = 0,57735 geklemmt, und die
	// Die Lagrange-Gewichte (4-/3-Punkt, NICHT Catmull-Rom -- Textkorrektur 3ee1542 hier nachgezogen, Audit 16.09.2026 Befund C-N4)
	// tragen je Achse hoechstens 1,25 in der Summe der Betraege, im 2D-Produkt
	// also 1,5625. Damit ist |v| <= 1,5625*0,57735 = 0,9021 < 1,0. Es ist eine INVARIANTENZUSICHERUNG,
	// kein Messinstrument: sie feuert erst, wenn jemand die Klemme oder die Lift-Gewichte aendert.
	// Slot 215 ist der Besuchszaehler dazu -- ohne ihn beweist die Null in 214 nichts, genau wie bei
	// 210/211 und 212/213 (dort war er von Anfang an da, hier fehlte er; im selben Block, sieben
	// Zeilen auseinander -- gefunden von zwei Pruefern unabhaengig).
	if(hits[215]<0xF0000000u) atomic_inc(&hits[215]); // Besuche des Lift-Schreibpfads
	if(fabs(v[1])>=1.0f||fabs(v[2])>=1.0f||fabs(v[3])>=1.0f) { if(hits[214]<0xF0000000u) atomic_inc(&hits[214]); return; }
)+"#ifdef RHO_RAND"+R(
	{ const uxx rr_ = rr_idx((uxx)n); // ★ C2b: Sentinel nur zaehlen (Slot 216, Soll 0), nie schreiben -- und KEIN return, die u-Schreibvorgaenge bleiben
	  if(rr_<(uxx)def_RR_N) store_rho(rho, rr_, v[0]); else if(hits[216]<0xF0000000u) atomic_inc(&hits[216]); }
)+"#else"+R(
	store_rho(rho, n, v[0]);
)+"#endif"+R( // RHO_RAND
	store_u(u, n, v[1]);
	store_u(u, def_N+(ulong)n, v[2]);
	store_u(u, 2ul*def_N+(ulong)n, v[3]);
} // drive_boundary_cubic_lift()

// ★ P9c N2F-SCHALE (Heiko-Idee): near -> far Schalen-RUECKKOPPLUNG. Die Hinkopplung oben ist
// EINWEG (Fernfeld diktiert dem Nahfeld die Raender); das Fernfeld rechnet das Fahrzeug aber nur
// als 32-mm-Treppenkoerper (+15,5 % Verdraengung, Census fc0efced/fc0edcf) und praegt dem Nahfeld
// damit ein zu grobes Druckfeld auf. P9c: auf einer SCHALE um die Fahrzeug-BBox des GROBGITTERS
// (5 Flaechen, z- entfaellt; Flaechen-Maske im Setup) wird das grobe Feld post-stream mit dem
// BLOCKGEMITTELTEN Nahfeld-u relaxiert: u_neu = (1-alpha)*u_far + alpha*u_near, f = f_eq(rho_lokal,
// u_neu) -- druckerhaltend (rho bleibt das LOKALE, Muster boden_eq/einlass_eq). KEIN voller
// Feedback-Overlap (2026-06-14-Lehre: dezimierte Nachlaufdaten wirkten als Barriere) -- die Schale
// liegt bewusst NAH am Koerper, wo das Nahfeld die bessere Verdraengung kennt, nicht im Nachlauf-
// scherfeld der Kopplungsebenen.
//
// schale_extract: laeuft auf BEIDEN Gittern. mittel=1 (Nahfeld): Blockmittel ueber ratio^3
// Feinzellen um den Deckungspunkt, NUR Fluid (TYPE_S/E uebersprungen, TYPE_MS zaehlt als Fluid --
// MS-Guard-Lehre 2f705ba); 0 Fluidzellen -> NaN-Marker (Host-Census beim Listenbau sagt an, wie
// viele Zellen das trifft; der Blend ueberspringt sie per Bit-Test). mittel=0 (Fernfeld/Waechter):
// Punktwert des u-FELDS der Schalenzelle. Die Liste enthaelt DIREKT die Zellindizes (fein:
// Deckungspunkte, grob: Schalenzellen) -- keine Offset-Rechnerei im Kernel.
// ★ Signatur-Abweichung vom Plan: flags ZUSAETZLICH gebunden -- ohne flags ist "NUR ueber Fluid"
// nicht entscheidbar (der Plan verlangt beides, das Fluid-Kriterium gewinnt).
// Fenster-Konvention: Offsets -ratio/2 .. ratio-ratio/2-1 je Achse (bei ratio=4: -2..+1) -- das
// Blockzentrum liegt eine HALBE Feinzelle unter dem Deckungspunkt (2 mm bei dx_f=4mm); fuer eine
// Relaxationsquelle unerheblich, aber deklariert.
)+R(kernel void schale_extract(const global velxx* u, const global uchar* flags, const global uint* liste, const uint n, const uint ratio, const uint mittel, global float* out) {
	const uint gid = get_global_id(0);
	if(gid>=n) return;
	const uxx c = (uxx)liste[gid];
	if(mittel==0u) { // Punktwert (Waechter auf dem Grobgitter): u-FELD-Wert der Schalenzelle
		out[3u*gid   ] = load_u(u, (ulong)c);
		out[3u*gid+1u] = load_u(u, def_N+(ulong)c);
		out[3u*gid+2u] = load_u(u, 2ul*def_N+(ulong)c);
		return;
	}
	const uint x = (uint)(c%(uxx)def_Nx), y = (uint)((c/(uxx)def_Nx)%(uxx)def_Ny), z = (uint)(c/((uxx)def_Nx*(uxx)def_Ny));
	float sx=0.0f, sy=0.0f, sz=0.0f; uint cnt=0u;
	const int r = (int)ratio, o0 = -(r/2); // Fenster [-r/2, r-r/2), s. Kopfkommentar
	for(int dz=o0; dz<o0+r; dz++) for(int dy=o0; dy<o0+r; dy++) for(int dx=o0; dx<o0+r; dx++) {
		const int xx=(int)x+dx, yy=(int)y+dy, zz=(int)z+dz;
		if(xx<0||yy<0||zz<0||xx>=(int)def_Nx||yy>=(int)def_Ny||zz>=(int)def_Nz) continue; // defensiv -- der Setup-Check haelt die Schale >=2 Grobzellen von den Entnahmeebenen fern
		const uxx nn = (uxx)xx+((uxx)yy+(uxx)zz*(uxx)def_Ny)*(uxx)def_Nx;
		const uchar bo = flags[nn]&TYPE_BO;
		if(bo==TYPE_S||bo==TYPE_E) continue; // nur Fluid; TYPE_MS wird MITGEZAEHLT (MS-Guard-Lehre)
		sx += load_u(u, (ulong)nn);
		sy += load_u(u, def_N+(ulong)nn);
		sz += load_u(u, 2ul*def_N+(ulong)nn);
		cnt++;
	}
	if(cnt==0u) { // kein Fluid im Block -> NaN-Marker, der Blend ueberspringt die Zelle (Bit-Test)
		const float nanm = as_float(0x7FC00000u);
		out[3u*gid]=nanm; out[3u*gid+1u]=nanm; out[3u*gid+2u]=nanm; return;
	}
	const float inv = 1.0f/(float)cnt;
	out[3u*gid]=sx*inv; out[3u*gid+1u]=sy*inv; out[3u*gid+2u]=sz*inv;
} // schale_extract()

// schale_blend: laeuft NUR auf dem Fernfeld (das Setup haelt s_schale_alpha des Nahfelds explizit
// auf 0; Wirkpfad-Endnachweis Slot 22 fern>0/nah==0). Range = n (INDEXLISTE, nicht def_N).
// Guards: TYPE_S/E return, TYPE_MS wird BEHANDELT (MS-Guard-Lehre 2f705ba); NaN-Bit-Test auf
// unear (Zelle ohne gueltiges Nah-Mittel bleibt unangetastet -- Census dazu macht der Host beim
// Listenbau, KEIN eigener Diag-Slot).
// ★ GRADIENT-BLEND (Heiko-Slice-Befund c646253): je Zelle wirkt a = alpha*gewicht[gid] -- der
// Host baut die Gewichte (Lagen-Gradient innen 1 -> aussen 1/N), der Kernel bleibt dumm.
// modus: 0 = EQ-Arm (Altverhalten: f = f_eq(rho_l, u_blend)); Bit 0 (CFD_N2F_SCHALE_FNEQ) =
// FNEQ-Arm: der Nichtgleichgewichtsanteil der Zelle wird ERHALTEN, f = f_eq(u_blend) + (f_true -
// f_eq(u_lokal)); 2 = IDENT-Debug-Arm: store_f(f_true) = exaktes No-Op (Paritaetsbeweis der
// Paarung, s. u.).
// ★★ PAARUNGS-HERLEITUNG (XL-B8, aus load_f/store_f abgeleitet): stream_collide(t) hat f_post[i]
// (ungerade i) nach (j[i], Slot t%2?i+1:i) und f_post[i+1] nach (n, Slot t%2?i:i+1) geschrieben.
// Der post-stream-re-load_f(t) liest fhn[i] von (n, t%2?i:i+1) = f_post[i+1] und fhn[i+1] von
// (j[i], t%2?i+1:i) = f_post[i] -- also fhn[i] = f_post[pair(i)] mit pair(0)=0, pair(ungerade i)=
// i+1, pair(gerade i)=i-1 (daher u exakt negiert, rho invariant). store_f(t) schreibt an EXAKT
// die Slots, aus denen f_post kam -- store_f(f_true) mit f_true[i]=fhn[pair(i)] legt jeden Wert
// bitgleich zurueck (fpxx->float->fpxx ist verlustfrei): das IDENT-No-Op. Damit ist store_f im
// TRUE-Frame adressiert und der EQ-Arm (store feq(u_blend), u_blend physikalisch) konsistent.
)+R(kernel void schale_blend(global fpxx* fi, const global uchar* flags, const ulong t, const float alpha, const global uint* liste, const uint n, const global float* unear, const global float* gewicht, const uint modus, volatile global uint* diag TS_P) {
	const uint gid = get_global_id(0);
	if(gid>=n) return;
	const uxx nn = (uxx)liste[gid];
)+"#ifdef SPARSE_TILES"+R(
	if(is_dead_tile(nn, tile_slot)) return; // XL-Audit B1; das Fernfeld faehrt im dd-Fall ohne Tiling, der Guard kostet dort nichts
)+"#endif"+R( // SPARSE_TILES
	const uchar bo = flags[nn]&TYPE_BO;
	if(bo==TYPE_S||bo==TYPE_E) return; // TYPE_MS ist FLUID und wird BEHANDELT (MS-Guard-Lehre 2f705ba)
	const float unx=unear[3u*gid], uny=unear[3u*gid+1u], unz=unear[3u*gid+2u];
	if((as_uint(unx)&0x7F800000u)==0x7F800000u||(as_uint(uny)&0x7F800000u)==0x7F800000u||(as_uint(unz)&0x7F800000u)==0x7F800000u) return; // NaN/Inf-Bit-Test (isfinite ist unter -cl-finite-math-only toter Code, Gross-Audit M)
	if(t%def_zaehl_takt==0ul) atomic_inc(&diag[22]); // Wirkpfad-Nachweis IM Binary (Iron Rule 3), Slot 22
	uxx j[def_velocity_set]; neighbors(nn, j);
	float fhn[def_velocity_set]; load_f(nn, fhn, fi, j, t TS_A);
	float rho_l, uxm, uym, uzm; calculate_rho_u(fhn, &rho_l, &uxm, &uym, &uzm);
)+"#if defined(KLEMM_BILANZ)&&defined(RHO_CLAMP)"+R(
	if((rho_l<=RHO_CLAMP_MIN||rho_l>=RHO_CLAMP_MAX)&&diag[269]<0xF0000000u) atomic_inc(&diag[269]); // ★ 15.09.2026 Klemmen S0d: Klemme in schale_blend, nur gezaehlt.
	// ★ BERICHTIGT Audit-Schleife 16.09.2026 (Befund A-N1): "massenerhaltend" gilt fuer ZWEI der drei Arme. FNEQ (Modus 1) speichert
	// f_eq(rho_c,u_blend) + f_true - f_eq(rho_c,u_lok) -> Summe = rho_roh, rho_c kuerzt sich heraus; IDENT (Modus 2) speichert f_true.
	// Der EQ-Arm (Modus 0, CFD_N2F_SCHALE_FNEQ=0, seit 22.08. nicht mehr Standard) speichert dagegen f_eq(rho_c): die Zellmasse springt
	// auf den GEKLEMMTEN Wert, und dieses drho wird nirgends gebucht ([267]/[268] gibt es nur in boden_eq/einlass_eq). Der Host warnt dort.
)+"#endif"+R( // KLEMM_BILANZ
	// ★★ XL-B8, hier TRAGEND (anders als in boden_eq/einlass_eq, die u verwerfen durften):
	// post-stream-load liest die Esoteric-Pull-Paare VERTAUSCHT -- calculate_rho_u liefert damit
	// u EXAKT NEGIERT (rho ist invariant). Das lokale u ist also u_lokal = -(uxm,uym,uzm).
	// Wer die drei Minuszeichen entfernt, blendet gegen -u_far und kippt die Schale in eine
	// Gegenstrom-Quelle -- der u-Negations-Nachweis im Setup (mittleres Schalen-u_x gegen u_inf)
	// und der Waechter schlagen dann beide an.
	const float ulx=-uxm, uly=-uym, ulz=-uzm;
	const float a = alpha*gewicht[gid]; // Gradient: Zellgewicht aus dem Host-Listenbau (Lagen-Rampe innen 1 -> aussen 1/N; x+ skalierbar)
	const float ux2=(1.0f-a)*ulx+a*unx, uy2=(1.0f-a)*uly+a*uny, uz2=(1.0f-a)*ulz+a*unz;
	float feq[def_velocity_set]; calculate_f_eq(rho_l, ux2, uy2, uz2, feq);
	if(modus==0u) { // EQ-Arm (Altverhalten; mit gewicht==1.0 bitidentisch zum alten alpha-Pfad: a = alpha*1.0f ist exakt)
		store_f(nn, feq, fi, j, t TS_A);
		return;
	}
	float ftrue[def_velocity_set]; // Paarung EXAKT gegen die EsoPull-Konvention (Herleitung im Kopfkommentar): pair(0)=0, pair(2k-1)=2k, pair(2k)=2k-1
	ftrue[0] = fhn[0];
	for(uint i=1u; i<def_velocity_set; i+=2u) { ftrue[i]=fhn[i+1u]; ftrue[i+1u]=fhn[i]; }
	// ★★ PAARUNGS- UND MOMENTEN-BEWEIS (Ersatz fuer den Paritaetszaehler, 2026-08-22 mittags).
	// WARUM ER GEBRAUCHT WIRD: der urspruengliche Zaehler weiter unten vergleicht feq[i] gegen
	// ftrue[i], NACHDEM feq[i] += ftrue[i]-feq_loc[i] gerechnet wurde. Bei a==0 ist feq==feq_loc
	// bitgleich, der Test lautet also x + (y-x) == y fuer BELIEBIGE x und y. Er ist damit
	// strukturell blind gegen genau die zwei Fehler, die er laut Kommentar ausschliessen sollte:
	// eine falsche Paarung schriebe ein anderes y (kuerzt sich weg), eine falsche Momentenrechnung
	// ein anderes x (kuerzt sich ebenfalls weg). Gefunden vom Pruefagenten 2026-08-22; das
	// "BESTANDEN" vom selben Morgen belegte fast nichts.
	//
	// DIESER TEST IST UNABHAENGIG: er rechnet die Momente aus ftrue NEU und stellt sie gegen die
	// aus fhn. Die Behauptung, die der ganze Arm traegt, lautet: post-stream-load liest die
	// EsoPull-Paare vertauscht, also ist rho invariant und u exakt negiert. Ist die Paarung falsch,
	// ist ftrue eine andere Permutation und die Momente negieren NICHT -- der Test schlaegt an.
	// Er laeuft in BEIDEN Armen, die ftrue benutzen (FNEQ und IDENT), und haengt nicht an alpha.
	if(t%def_zaehl_takt==0ul) {
		float rho_t, uxt, uyt, uzt; calculate_rho_u(ftrue, &rho_t, &uxt, &uyt, &uzt);
		const float su = fmax(fmax(fabs(uxm), fabs(uym)), fmax(fabs(uzm), 1e-4f)); // Bezug: groesste Komponente, Boden 1e-4 gegen Division durch ~0 in Staugebieten
		const float du = fmax(fmax(fabs(uxt-ulx), fabs(uyt-uly)), fabs(uzt-ulz))/su;
		const float dr = fabs(rho_t-rho_l)/fmax(fabs(rho_l), 1e-4f);
		const float dd = fmax(du, dr);
		if(dd>1.0e-5f) atomic_inc(&diag[25]);                                  // 1 je ZELLE (nicht je DDF): kein uint-Wickel
		atomic_max(&diag[26], convert_uint_sat(fmin(dd, 4.0f)*1.0e9f));        // Mass, gesaettigt statt undefiniert
	}
	if(modus==2u) { // IDENT-Debug-Arm: exaktes No-Op -- harter Paritaetsbeweis der Paarung (Abnahme d)
		store_f(nn, ftrue, fi, j, t TS_A);
		return;
	}
	// FNEQ-Arm (modus&1): f_neu = f_eq(u_blend) + (f_true - f_eq(rho_l, u_lokal)) -- feq_loc mit
	// DENSELBEN negierten Momenten (ulx,uly,ulz) wie der Blend, sonst waere f_neq kein reiner
	// Nichtgleichgewichtsanteil (bei exakter Equilibrium-Zelle MUSS f_true - feq_loc = 0 sein).
	float feq_loc[def_velocity_set]; calculate_f_eq(rho_l, ulx, uly, ulz, feq_loc);
	for(uint i=0u; i<def_velocity_set; i++) feq[i] += ftrue[i]-feq_loc[i];
	// ★ RUNDUNGSZAEHLER (hiess bis 2026-08-22 mittags "Paritaetszaehler"). Er misst, WAS ER MISST:
	// die Gleitkomma-Rundung von x + (y-x) gegen y, und dass alpha als exakte 0 im Kernel ankommt.
	// Er misst NICHT die Paarung und NICHT die Momentenrechnung -- x kuerzt sich algebraisch heraus
	// (Pruefagenten-Befund 2026-08-22). Der echte Beweis dafuer steht oben, Slots 25/26.
	// WARUM GERAETEINTERN und nicht per Dateivergleich: im Doppel-Domaenen-Fall ist ein
	// Bitvergleich zweier Laeufe UNMOEGLICH -- die blosse Aktivierung der Rueckkopplung zieht
	// schale_extract_u auf dem Nahfeld ins Kopplungsfenster, und das Nahfeld ist ueber po_mean
	// feldwirksam nichtdeterministisch. Gemessen am 2026-08-22: zwei WORTGLEICHE IDENT-Laeufe
	// liefern verschiedene interface_druck.csv. Der Zaehler haengt dagegen an nichts als der
	// Arithmetik dieser Zelle.
	if(t%def_zaehl_takt==0ul) { // Zahl der Abweichungen UND ihr Mass (groesste ULP-Distanz), denn "ungleich"
		// allein unterscheidet nicht zwischen einem gebrochenen Paar und der Rundung von a+(b-a).
		uint abw=0u, ulp_max=0u;
		for(uint i=0u; i<def_velocity_set; i++) if(feq[i]!=ftrue[i]) {
			abw++;
			// RELATIVE Abweichung, nicht float32-ULP: gespeichert wird in FP16S (defines.hpp:27 -- hier
			// stand FP16C, berichtigt 21.09.2026; beide 2-Byte-Formate), dessen Aufloesung
			// bei ~2^-11 = 4,9e-4 liegt. Zwei float32-Werte, die sich um 1000 ULP (= 1,2e-4 relativ)
			// unterscheiden, landen im 16-Bit-Format auf DEMSELBEN Wert -- ein Vergleich in float32-
			// ULP misst also eine Genauigkeit, die das Feld gar nicht traegt.
			const float rel = fabs(feq[i]-ftrue[i])/fmax(fabs(ftrue[i]), 1e-6f);
			const uint sk = convert_uint_sat(fmin(rel, 4.0f)*1.0e9f); // gesaettigt: float->uint ausserhalb des Bereichs ist in OpenCL implementierungsdefiniert, und im Diagnosezweig (alpha>0) wird rel gross
			ulp_max = sk>ulp_max ? sk : ulp_max;
		}
		if(abw>0u) { atomic_inc(&diag[23]); atomic_max(&diag[24], ulp_max); } // 1 je ZELLE: mit 19 je Zelle wickelte der uint-Slot bei Default-N und Wake-Kasten schon nach ~1 s (Pruefagent-Befund 4)
	}
	store_f(nn, feq, fi, j, t TS_A);
} // schale_blend()

)+"#ifdef FORCE_FIELD"+R(
)+R(kernel void update_force_field(const global fpxx* fi, const global uchar* flags, const ulong t, global float* F, const global uint* f_maske, const global velxx* u, global uint* hits TS_P) { // calculate force from the fluid on solid boundaries from fi directly
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute update_force_field() on halo
)+"#ifdef SPARSE_TILES"+R(
	// FORK: Zellen in toten Tiles duerfen fi NIE anfassen. cell_base() gibt fuer sie defensiv 0
	// zurueck -- ohne diesen Ausstieg landen ihre Schreibzugriffe also in Slot 0 und ueberschreiben
	// die Daten einer echten aktiven Tile. Genau daran ist der erste T=8-Lauf divergiert (Cd 18.4).
	if(is_dead_tile(n, tile_slot)) return;
)+"#endif"+R(
	if((flags[n]&TYPE_BO)!=TYPE_S) return; // only continue for solid boundary cells
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	// FORK: nur Solidzellen mit MINDESTENS EINEM Fluidnachbarn rechnen. Die Momentum-Exchange-Kraft
	// entsteht dort, wo Verteilungen zurueckprallen -- eine Zelle tief im Koerperinneren hat keinen
	// Fluidnachbarn, liefert null und kostet nur Zeit. Beim Block-Tiling ist es schlimmer als nutzlos:
	// ihre Nachbarn liegen in toten Tiles, cell_base() liefert dafuer den Papierkorb-Slot 0, und dessen
	// Inhalt ist weder null noch definiert -- die Zelle traegt dann eine erfundene Kraft bei, die im
	// dichten Pfad nicht existiert. Genau daran haette die behauptete Bit-Neutralitaet scheitern koennen.
	bool has_fluid_neighbor = false;
	for(uint i=1u; i<def_velocity_set; i++) has_fluid_neighbor = has_fluid_neighbor || (flags[j[i]]&TYPE_BO)!=TYPE_S;
)+"#ifdef F_LISTE"+R(
	// ★ 03.09.: unter der Markerliste hat eine Solidzelle OHNE Fluidnachbarn konstruktiv keinen Slot.
	// Der Nullschreiber ist dort gegenstandslos -- und er WAR die Quelle von 1,85 Mrd. Treffern auf
	// Slot 77 im ersten 8-mm-Listenlauf (der Waechter hat also richtig gezaehlt, nur an harmlosen
	// Zellen). Genau das stand in der Vorpruefung zu diesem Posten: "In Form B muss daraus ein return
	// werden." Gelesen, nicht umgesetzt -- hier nachgeholt.
	if(!has_fluid_neighbor) return;
)+"#else"+R(
	if(!has_fluid_neighbor) { store3_F(F, f_maske, n, (float3)(0.0f, 0.0f, 0.0f), hits); return; }
)+"#endif"+R( // F_LISTE
	float fhn[def_velocity_set]; // local DDFs
	load_f(n, fhn, fi, j, t TS_A); // perform streaming (part 2)
	// ★★ 2026-08-25, Pruefbefund 2-A. Vorher: calculate_rho_u ueber ALLE 19 Richtungen, also auch ueber
	// Links, deren Streaming-Ursprung selbst Solid ist. Fuer die fuehrt NIEMAND je store_f aus -- der
	// Slot behaelt den Wert aus initialize(). Bei ruhender Wand ist das feq(1,0), in Stoerform exakt 0,
	// deshalb ist es nie aufgefallen. Bei MITBEWEGTER Wand ist es feq(1,u_w) und NICHT null: an der
	// ebenen Fahrbahn sind 13 der 18 Links tot und tragen +-5*u_w/3 -- mit der Paritaet des Zeitschritts
	// oszillierend, drei Groessenordnungen ueber tau_w. Meine erste Fassung addierte nur den fehlenden
	// Bewegtwand-Term und liess diese vier Fuenftel stehen. Jetzt laufen BEIDE Summanden ueber DIESELBE
	// Linkmenge: nur Links, die wirklich an Fluid zurueckgeworfen werden.
	//   F = Sum_{Ursprung Fluid} c_i * (2 f_i - 6 w_i (c_i . u_w))   (Krueger S.180, rho_w = 1)
	// Fuer ruhende Waende ist das Ergebnis physikalisch identisch (die toten Links tragen 0); NICHT
	// bitgleich, weil sich die Summationsreihenfolge gegenueber calculate_rho_u aendert.
	float Fx=0.0f, Fy=0.0f, Fz=0.0f;
)+"#ifdef MOVING_BOUNDARIES"+R(
	const float uwx=load_u(u, n), uwy=load_u(u, def_N+(ulong)n), uwz=load_u(u, 2ul*def_N+(ulong)n);
	const bool bewegt = (uwx!=0.0f||uwy!=0.0f||uwz!=0.0f);
)+"#endif"+R( // MOVING_BOUNDARIES
	for(uint i=1u; i<def_velocity_set; i++) {
		const uint ib = (i%2u==1u) ? i+1u : i-1u; // Streaming-Ursprung von fhn[i] ist j[opposite(i)]
		if((flags[j[ib]]&TYPE_BO)==TYPE_S) continue; // toter Link: nie beschrieben, traegt den initialize()-Wert
		const float cix=c(i), ciy=c(def_velocity_set+i), ciz=c(2u*def_velocity_set+i);
		float m = 2.0f*fhn[i];
)+"#ifdef MOVING_BOUNDARIES"+R(
		if(bewegt) m = fma(-6.0f*w(i), cix*uwx+ciy*uwy+ciz*uwz, m); // Bewegtwand-Anteil B_i
)+"#endif"+R( // MOVING_BOUNDARIES
		Fx = fma(m, cix, Fx); Fy = fma(m, ciy, Fy); Fz = fma(m, ciz, Fz);
	}
)+"#ifdef MOVING_BOUNDARIES"+R(
	if(bewegt&&hits[59]<0xF0000000u) atomic_inc(&hits[59]); // Wirkpfad, saettigend: update_force_field laeuft NICHT jeden Schritt, eine t%100-Gatung waere im ungeeigneten Takt strukturell null
)+"#endif"+R( // MOVING_BOUNDARIES
	store3_F(F, f_maske, n, (float3)(Fx, Fy, Fz), hits); // FORK: bbox-bewusst // 2x, weil fi an Solidzellen zurueckgeworfen werden
} // update_force_field()
)+R(kernel void reset_force_field(global float* F, const global uint* f_maske, global uint* hits) { // reset force field
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	if(n>=(uxx)def_N) return; // execute reset_force_field() also on halo
	store3_F(F, f_maske, n, (float3)(0.0f, 0.0f, 0.0f), hits); // FORK: bbox-bewusst
} // reset_force_field()
)+R(kernel void object_center_of_mass(const global uchar* flags, const uchar flag_marker, volatile global float* object_sum) {
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	const uint lid = get_local_id(0); // local memory reduction of cl_workgroup_size:1
	local float3 cache[cl_workgroup_size];
	local uint cells[cl_workgroup_size];
	const uint is_part_of_object = (uint)(n<(uxx)def_N&&flags[n]==flag_marker);
	cache[lid] = is_part_of_object ? position(coordinates(n)) : (float3)(0.0f, 0.0f, 0.0f);
	cells[lid] = is_part_of_object;
	barrier(CLK_GLOBAL_MEM_FENCE);
	for(uint s=1u; s<cl_workgroup_size; s*=2u) {
		if(lid%(2u*s)==0u) {
			cache[lid] += cache[lid+s];
			cells[lid] += cells[lid+s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	const uint local_cells = cells[0];
	if(lid==0u&&local_cells>0u) { // global memory reduction with atomic addition of local_sum
		const float3 local_sum = cache[0];
		atomic_add_f(&object_sum[0], local_sum.x);
		atomic_add_f(&object_sum[1], local_sum.y);
		atomic_add_f(&object_sum[2], local_sum.z);
		atomic_add((volatile global uint*)&object_sum[3], local_cells);
	}
} // object_center_of_mass()
)+R(kernel void object_force(const global float* F, const global uint* f_maske, const global uchar* flags, const uchar flag_marker, global float* of_part) {
	// ★★ 2026-08-25, DETERMINISMUS. Vorher: je Arbeitsgruppe ein atomic_add_f auf object_sum --
	// die Additionsreihenfolge haengt an der Scheduling-Reihenfolge, Gleitkomma-Addition ist nicht
	// assoziativ, also lieferten bitgleiche Laeufe verschiedene Cd/Cz in den letzten Stellen. Das
	// war die zweite (berichtende) der beiden Divergenzquellen; die erste (po_mean) fiel mit 849b14f.
	// Jetzt: FESTE Gittergroesse mit Grid-Stride-Schleife -> feste Zahl Teilsummen, feste Summations-
	// reihenfolge in object_force_final. Nebenbei entfaellt der Atomic-Verkehr komplett.
	const uint lid = get_local_id(0);
	const ulong gid = (ulong)get_global_id(0), gs = (ulong)get_global_size(0); // ★ Pruefbefund 1-b: als uxx wickelt die Schleife im 65536-Fenster unter der uxx-Grenze zur Endlosschleife
	local float3 cache[cl_workgroup_size];
	float3 s = (float3)(0.0f, 0.0f, 0.0f);
	for(ulong n=gid; n<(ulong)def_N; n+=gs) if(flags[(uxx)n]==flag_marker) s += load3_F(F, f_maske, (uxx)n); // feste Reihenfolge je Work-Item
	cache[lid] = s;
	barrier(CLK_LOCAL_MEM_FENCE); // ★ war CLK_GLOBAL_MEM_FENCE -- der falsche Speicher fuer ein local-Array
	for(uint st=1u; st<cl_workgroup_size; st*=2u) {
		if(lid%(2u*st)==0u) cache[lid] += cache[lid+st];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid==0u) { const uint g=get_group_id(0); of_part[3u*g]=cache[0].x; of_part[3u*g+1u]=cache[0].y; of_part[3u*g+2u]=cache[0].z; }
} // object_force()
)+R(kernel void object_force_zband(const global float* F, const global uint* f_maske, const global uchar* flags, const uchar flag_marker, const uint z_lo, const uint z_hi, global float* of_part) {
	// FORK Kraft-Zerlegung (CFD_KRAFT_ZBAND): woertliche Kopie von object_force mit z-Band-Praedikat
	// [z_lo,z_hi). coordinates() ist DOMAENENLOKAL -- der LBM-Wrapper erzwingt D=1.
	const uint lid = get_local_id(0);
	const ulong gid = (ulong)get_global_id(0), gs = (ulong)get_global_size(0); // ★ Pruefbefund 1-b: als uxx wickelt die Schleife im 65536-Fenster unter der uxx-Grenze zur Endlosschleife
	local float3 cache[cl_workgroup_size];
	float3 s = (float3)(0.0f, 0.0f, 0.0f);
	for(ulong n=gid; n<(ulong)def_N; n+=gs) if(flags[(uxx)n]==flag_marker&&coordinates((uxx)n).z>=z_lo&&coordinates((uxx)n).z<z_hi) s += load3_F(F, f_maske, (uxx)n);
	cache[lid] = s;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(uint st=1u; st<cl_workgroup_size; st*=2u) {
		if(lid%(2u*st)==0u) cache[lid] += cache[lid+st];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid==0u) { const uint g=get_group_id(0); of_part[3u*g]=cache[0].x; of_part[3u*g+1u]=cache[0].y; of_part[3u*g+2u]=cache[0].z; }
} // object_force_zband()
)+R(kernel void object_force_final(const global float* of_part, const uint n_groups, global float* object_sum) {
	if(get_global_id(0)!=0u) return; // eine feste, serielle Summationsreihenfolge -- das ist der ganze Zweck
	float sx=0.0f, sy=0.0f, sz=0.0f;
	for(uint g=0u; g<n_groups; g++) { sx+=of_part[3u*g]; sy+=of_part[3u*g+1u]; sz+=of_part[3u*g+2u]; }
	object_sum[0]=sx; object_sum[1]=sy; object_sum[2]=sz;
} // object_force_final()
)+R(kernel void kraft_facetten_gpu(const global float* F, const global uint* f_maske, const global uint* kf_liste, const uint kf_N, const global uint* fac_idx, const global uint* fac_tau_n, const global float* fac_geo, const uint fac_on, const uint z_per, global float* kf_psum, global uint* kf_pcnt) {
	// FORK kraft_facetten-GPU: Druckanteil des Facetten-Cd-Pfads ohne Host-F-Transfer. Range =
	// Markerzellen-Indexliste (der Host baut sie in der Dreifachschleifen-Scan-Reihenfolge der
	// F-BBox). Klassifikation voll/projiziert/unklar und Projektion AUSDRUCKSGLEICH zum Host-Pfad
	// kraft_facetten (setup.cpp), Rechnung float; die double-Endsumme bildet der Host in fester
	// Gruppenreihenfolge. KEINE globalen float-Atomics: Workgroup-Baum-Reduktion (Muster
	// po_reduce_mean), lid==0 schreibt atomikfrei die exklusiven Slots kf_psum[3*Gruppe+0..2]
	// (px,py,pz) und kf_pcnt[3*Gruppe+0..2] (voll,proj,unklar).
	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	// D3Q19-Richtungen 1..18 in EXAKT der Host-Reihenfolge (setup.cpp FZ_C) -- NICHT die
	// velocity_set-Reihenfolge dieses Kernels.
	const int fzc[19][3] = {{0,0,0},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
		{1,1,0},{-1,-1,0},{1,0,1},{-1,0,-1},{0,1,1},{0,-1,-1},{1,-1,0},{-1,1,0},{1,0,-1},{-1,0,1},{0,1,-1},{0,-1,1}};
	float px=0.0f, py=0.0f, pz=0.0f; uint cv=0u, cq=0u, cu=0u; // Beitrag dieses Work-Items (Padding-Items bleiben 0)
	if(gid<kf_N) {
		const uxx n = (uxx)kf_liste[gid];
		const uint3 xyz = coordinates(n);
		uxx fbi;
		if(f_bbox(n, &fbi)) { // Markerzellen liegen konstruktiv in der F-BBox; der Test bleibt als Waechter
			const float3 Fv = load3_F(F, f_maske, n); const float Fx=Fv.x, Fy=Fv.y, Fz=Fv.z; // ★ 03.09.: statt handgeschriebenem BBox-Zugriff jetzt UEBER load3_F -- die F-Markerliste haette den Zwilling sonst stumm falsch indiziert (die Duplikat-Falle, die object_torque am 08.08. schon einmal ausgeloest hat)
			float nxm=0.0f, nym=0.0f, nzm=0.0f; bool kontaminiert=false;
			if(fac_on!=0u) for(uint i=1u; i<19u; i++) { // fac_on==0: Nachbarschleife entfaellt, alles zaehlt als voll (object_force-Semantik auf der Liste)
				// x/y-Wrap, z-Klemme bzw. z_per-Wrap und F-BBox-Clip AUSDRUCKSGLEICH zu setup.cpp (def_FB* statt D->fb*)
				const int zn0=(int)xyz.z+fzc[i][2]; if(z_per==0u&&(zn0<0||zn0>=(int)def_Nz)) continue;
				const int zn=z_per!=0u?(int)((zn0%(int)def_Nz+(int)def_Nz)%(int)def_Nz):zn0;
				const uint xn=(uint)((((int)xyz.x+fzc[i][0])%(int)def_Nx+(int)def_Nx)%(int)def_Nx);
				const uint yn=(uint)((((int)xyz.y+fzc[i][1])%(int)def_Ny+(int)def_Ny)%(int)def_Ny);
				if(xn<def_FBX0||yn<def_FBY0||(uint)zn<def_FBZ0||xn>=def_FBX0+def_FBNX||yn>=def_FBY0+def_FBNY||(uint)zn>=def_FBZ0+def_FBNZ) continue;
				const ulong fbi2=(ulong)(xn-def_FBX0)+((ulong)(yn-def_FBY0)+(ulong)((uint)zn-def_FBZ0)*(ulong)def_FBNY)*(ulong)def_FBNX;
				const uint fid = fac_fid(fac_idx, (uxx)fbi2);
				if(fid==0xFFFFFFFFu||fac_tau_n[fid]==0u) continue;
				kontaminiert=true;
				nxm+=fac_geo[8ul*(ulong)fid]; nym+=fac_geo[8ul*(ulong)fid+1ul]; nzm+=fac_geo[8ul*(ulong)fid+2ul];
			}
			if(!kontaminiert) { px=Fx; py=Fy; pz=Fz; cv=1u; }
			else {
				const float l = sqrt(nxm*nxm+nym*nym+nzm*nzm);
				if(l<0.5f) { px=Fx; py=Fy; pz=Fz; cu=1u; } // Gegennormalen (Spalt): konservativ voll -- Schwelle wie Host (l<0.5)
				else {
					const float nx2=nxm/l, ny2=nym/l, nz2=nzm/l;
					const float fn = Fx*nx2+Fy*ny2+Fz*nz2;
					px=fn*nx2; py=fn*ny2; pz=fn*nz2; cq=1u;
				}
			}
		}
	}
	local float cache_x[cl_workgroup_size], cache_y[cl_workgroup_size], cache_z[cl_workgroup_size];
	local uint cnt_v[cl_workgroup_size], cnt_q[cl_workgroup_size], cnt_u[cl_workgroup_size];
	cache_x[lid]=px; cache_y[lid]=py; cache_z[lid]=pz; cnt_v[lid]=cv; cnt_q[lid]=cq; cnt_u[lid]=cu;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(uint s=1u; s<cl_workgroup_size; s*=2u) { // Baum-Reduktion, Muster po_reduce_mean
		if(lid%(2u*s)==0u) {
			cache_x[lid]+=cache_x[lid+s]; cache_y[lid]+=cache_y[lid+s]; cache_z[lid]+=cache_z[lid+s];
			cnt_v[lid]+=cnt_v[lid+s]; cnt_q[lid]+=cnt_q[lid+s]; cnt_u[lid]+=cnt_u[lid+s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid==0u) { // OHNE Atomik: jede Gruppe schreibt exklusiv ihre drei Slots
		const uint g = get_group_id(0);
		kf_psum[3u*g]=cache_x[0]; kf_psum[3u*g+1u]=cache_y[0]; kf_psum[3u*g+2u]=cache_z[0];
		kf_pcnt[3u*g]=cnt_v[0];   kf_pcnt[3u*g+1u]=cnt_q[0];   kf_pcnt[3u*g+2u]=cnt_u[0];
	}
} // kraft_facetten_gpu()

)+R(kernel void sgs_gdiag(const global fpxx* fi, const global velxx* u, const global uchar* flags,
	const global uint* gd_zellen, const uint gd_N, global float* fac_gd, const ulong t,
	const float fx, const float fy, const float fz, const uint guo_an TS_P) {
	// ★★ g-DIAGNOSE (31.08.2026, Pruefagenten-Empfehlung "Messung statt Wette", ARBEITSLISTE Vorzeichen-
	// Einwand). Sparser Kernel ueber die Facettenzellenliste, laeuft NACH einem abgeschlossenen Schritt
	// (Host-Enqueue an der Chunk-/Sample-Kadenz). Fasst weder w noch f noch u an -- reine Messung.
	// Je Wandzelle wird akkumuliert:
	//  [0] |S|_FD    Scherratenbetrag aus dem u-FELD (zentrale Differenzen ueber die 6 Flaechennachbarn,
	//                Solid-Nachbar: u = 0, no-slip erster Ordnung) -- ein echtes Moment, IMMUN gegen
	//                Geistermoden des Wandmodells (apply_facette_imem schreibt nicht-hydrodynamische
	//                Populationen in fhn, und aus fhn baut der SUBGRID-Block seinen Tensor);
	//  [1] |S|_Pi    dieselbe Groesse aus Pi^neq = Sum cc(f-feq), AUSDRUCKSGLEICH zum SUBGRID-Block
	//                (inkl. Guo-Korrektur des zweiten Moments und selbstkonsistentem tau_eff) --
	//                das, was Smagorinsky heute an dieser Zelle wirklich sieht;
	//  [2] D_WALE    was WALE liefern WUERDE (Nicoud/Ducros-Operator auf dem vollen Gradienten);
	//  [3] D_Sigma   was Sigma liefern WUERDE (Singulaerwerte von g, Cardano);
	//  [4] |Omega|_FD Rotationsbetrag (reine Scherung: |Omega| == |S|; die Abweichung davon sagt,
	//                wie weit die Zelle von reiner Scherung entfernt ist);
	//  [5] Besuche, [6] Anzahl solider Flaechennachbarn (geometrisch konstant), [7] frei.
	// PARITAETSFESTIGKEIT: nach dem Streaming koennen die Linkpaare vertauscht liegen (boden_eq-Lehre:
	// "post-stream-load = Paare vertauscht -> u waere NEGIERT"). Sum cc f ist unter i<->ib INVARIANT
	// (c*c ist gerade), rho ebenfalls -- deshalb kommt u hier NICHT aus calculate_rho_u, sondern aus dem
	// u-Feld; das traegt zudem exakt den Guo-Halbschub, mit dem der SUBGRID-Block sein feq bildet.
	const uint gid = get_global_id(0);
	if(gid>=gd_N) return;
	const uxx n = (uxx)gd_zellen[gid];
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	// ---- voller Gradient g[i][a] = du_i/dx_a; Flaechennachbar-Paare (1,2)=(+x,-x), (3,4)=(+y,-y), (5,6)=(+z,-z)
	float g[3][3]; uint nsolid=0u;
	for(uint a=0u; a<3u; a++) {
		const uxx np=j[2u*a+1u], nm=j[2u*a+2u];
		const bool sp=(flags[np]&TYPE_BO)==TYPE_S, sm=(flags[nm]&TYPE_BO)==TYPE_S;
		nsolid += (uint)sp+(uint)sm;
		for(uint i=0u; i<3u; i++) {
			const float up_ = sp?0.0f:load_u(u, (ulong)i*def_N+(ulong)np);
			const float um_ = sm?0.0f:load_u(u, (ulong)i*def_N+(ulong)nm);
			g[i][a] = 0.5f*(up_-um_);
		}
	}
	float SS=0.0f, WW=0.0f;
	for(uint i=0u;i<3u;i++) for(uint a=0u;a<3u;a++) {
		const float Sia=0.5f*(g[i][a]+g[a][i]), Wia=0.5f*(g[i][a]-g[a][i]);
		SS=fma(Sia,Sia,SS); WW=fma(Wia,Wia,WW);
	}
	const float snorm_fd = sqrt(2.0f*SS), onorm_fd = sqrt(2.0f*WW);
	// ---- WALE (Sd = sym(g*g) - tr(g*g)/3 * I); x^1.5 = x*sqrt(x), x^2.5 = x*x*sqrt(x), x^1.25 = x*sqrt(sqrt(x))
	float gg[3][3];
	for(uint i=0u;i<3u;i++) for(uint a=0u;a<3u;a++) gg[i][a] = g[i][0]*g[0][a]+g[i][1]*g[1][a]+g[i][2]*g[2][a];
	const float trg2 = (gg[0][0]+gg[1][1]+gg[2][2])*(1.0f/3.0f);
	float SdSd=0.0f;
	for(uint i=0u;i<3u;i++) for(uint a=0u;a<3u;a++) {
		const float Sd = 0.5f*(gg[i][a]+gg[a][i]) - (i==a?trg2:0.0f);
		SdSd=fma(Sd,Sd,SdSd);
	}
	const float d_wale = (SdSd*sqrt(SdSd)) / (SS*SS*sqrt(SS) + SdSd*sqrt(sqrt(SdSd)) + 1e-30f);
	// ---- Sigma (Singulaerwerte von g = sqrt(Eigenwerte von G = g^T g), Cardano mit Klemmen)
	float G00=0.0f,G11=0.0f,G22=0.0f,G01=0.0f,G02=0.0f,G12=0.0f;
	for(uint i=0u;i<3u;i++) { G00=fma(g[i][0],g[i][0],G00); G11=fma(g[i][1],g[i][1],G11); G22=fma(g[i][2],g[i][2],G22);
		G01=fma(g[i][0],g[i][1],G01); G02=fma(g[i][0],g[i][2],G02); G12=fma(g[i][1],g[i][2],G12); }
	float l1,l2,l3;
	{	const float p1 = G01*G01+G02*G02+G12*G12;
		const float q  = (G00+G11+G22)*(1.0f/3.0f);
		if(p1<1e-30f) { // (nahezu) diagonal: Eigenwerte = Diagonale, absteigend sortieren
			l1=fmax(G00,fmax(G11,G22)); l3=fmin(G00,fmin(G11,G22)); l2=G00+G11+G22-l1-l3;
		} else {
			const float p2 = (G00-q)*(G00-q)+(G11-q)*(G11-q)+(G22-q)*(G22-q)+2.0f*p1;
			const float p  = sqrt(p2*(1.0f/6.0f));
			const float ip = 1.0f/fmax(p,1e-30f);
			const float B00=(G00-q)*ip, B11=(G11-q)*ip, B22=(G22-q)*ip, B01=G01*ip, B02=G02*ip, B12=G12*ip;
			float r = 0.5f*(B00*(B11*B22-B12*B12)-B01*(B01*B22-B12*B02)+B02*(B01*B12-B11*B02));
			r = clamp(r,-1.0f,1.0f);
			const float phi = acos(r)*(1.0f/3.0f);
			l1 = q+2.0f*p*cos(phi);
			l3 = q+2.0f*p*cos(phi+2.0943951f); // + 2*pi/3
			l2 = 3.0f*q-l1-l3;
		}
	}
	const float s1=sqrt(fmax(l1,0.0f)), s2=sqrt(fmax(l2,0.0f)), s3=sqrt(fmax(l3,0.0f));
	const float d_sigma = s1>1e-20f ? s3*(s1-s2)*(s2-s3)/(s1*s1) : 0.0f;
	// ---- Pi^neq ausdrucksgleich zum SUBGRID-Block (rho aus f -- paarinvariant; u aus dem Feld)
	float fhn[def_velocity_set];
	load_f(n, fhn, fi, j, t TS_A);
	float rhon, dux, duy, duz;
	calculate_rho_u(fhn, &rhon, &dux, &duy, &duz); // dux..duz VERWERFEN (Paritaetsfalle), nur rhon zaehlt
	const float uxn=load_u(u, n), uyn=load_u(u, def_N+(ulong)n), uzn=load_u(u, 2ul*def_N+(ulong)n);
	float feq[def_velocity_set];
	calculate_f_eq(rhon, uxn, uyn, uzn, feq);
	float Hxx=0.0f, Hyy=0.0f, Hzz=0.0f, Hxy=0.0f, Hxz=0.0f, Hyz=0.0f;
	for(uint i=1u; i<def_velocity_set; i++) {
		const float fneqi = fhn[i]-feq[i];
		const float cxi=c(i), cyi=c(def_velocity_set+i), czi=c(2u*def_velocity_set+i);
		Hxx=fma(cxi*cxi,fneqi,Hxx); Hyy=fma(cyi*cyi,fneqi,Hyy); Hzz=fma(czi*czi,fneqi,Hzz);
		Hxy=fma(cxi*cyi,fneqi,Hxy); Hxz=fma(cxi*czi,fneqi,Hxz); Hyz=fma(cyi*czi,fneqi,Hyz);
	}
	if(guo_an!=0u) { // Guo-Korrektur des zweiten Moments wie im SUBGRID-Block (SGS_GUO); an Facettenzellen
		// ist F +0.0f (F-NUR-SOLID), die konstante Volumenkraft fx..fz ist der ganze Beitrag.
		Hxx=fma(uxn,fx,Hxx); Hyy=fma(uyn,fy,Hyy); Hzz=fma(uzn,fz,Hzz);
		Hxy=fma(0.5f,uxn*fy+fx*uyn,Hxy); Hxz=fma(0.5f,uxn*fz+fx*uzn,Hxz); Hyz=fma(0.5f,uyn*fz+fy*uzn,Hyz);
	}
	const float Q = sq(Hxx)+sq(Hyy)+sq(Hzz)+2.0f*(sq(Hxy)+sq(Hxz)+sq(Hyz));
	const float tau0 = 1.0f/def_w;
	const float tau_eff = 0.5f*(tau0+sqrt(sq(tau0)+0.76421222f*sqrt(Q)/rhon)); // selbstkonsistent wie kernel-|S|-Formel
	const float snorm_pi = 3.0f*sqrt(2.0f*Q)/(2.0f*rhon*tau_eff); // S_ij = -3 Pi_ij/(2 rho tau) -> |S| = 3 sqrt(2 Q)/(2 rho tau)
	// ---- Akkumulation (racefrei: 1 Work-Item = 1 Facette)
	const ulong k8 = 8ul*(ulong)gid;
	fac_gd[k8]      += snorm_fd;
	fac_gd[k8+1ul]  += snorm_pi;
	fac_gd[k8+2ul]  += d_wale;
	fac_gd[k8+3ul]  += d_sigma;
	fac_gd[k8+4ul]  += onorm_fd;
	fac_gd[k8+5ul]  += 1.0f;
	fac_gd[k8+6ul]  += (float)nsolid;
} // sgs_gdiag()

)+R(kernel void sgs_fdwand)+"("+R(const global velxx* u, const global uchar* flags,
	const global uint* gd_zellen, const uint gd_N, global float* fac_wfd // ) {
)+"#ifdef SGS_SISM"+R(
	, const ulong t, global float* fac_sb, global uint* rho_clamp_hits, const uint sbar_out // ★ 07.09. SISM: Reihenfolge = add_parameters in alloc_facetten_domain (t, fac_sb, hits), VOR tile_slot. ★ 08.09. sbar_out: 0 = fac_wfd traegt w (Lage 1, Geistermoden-Fix), 1 = es traegt Sbar (Band -- dort ersetzt nichts das w, stream_collide zieht Sbar selbst ab)
)+"#endif"+R( // SGS_SISM
	TS_P
)+") {"+R( // sgs_fdwand() -- Kopfklammern als Strings AUSSERHALB von R() (Muster stream_collide): R() zaehlt Klammern; ein Splice zwischen "(" und ")" innerhalb EINES R-Blocks wird bis zur balancierten Klammer als TEXT stringifiziert -- so kam 07.09. abends ')+"#ifdef SGS_SISM"+R(' woertlich in den OpenCL-Quelltext (JIT -11 in ALLEN Armen, auch SISM=0)
	// ★★ SGS-GEISTERMODEN-FIX (CFD_SGS_FDWAND, 02.09.2026, Heiko-Go "korrigiere bitte das sgs";
	// Befunde B66/B69: das iMEM-Wandmodell schreibt nicht-hydrodynamische Populationen in fhn, und
	// Smagorinsky baut daraus seinen Tensor -- Pi/FD = 2,3-3,4 an anwendenden Wandzellen. WALE/Sigma
	// sind gemessen KEINE Loesung, SGS_WANDFREI kollabiert c_f um Faktor 35). Dieser Kernel berechnet
	// je Facettenzelle die RELAXATIONSRATE w aus dem GEISTERMODENFREIEN |S|_FD des u-Felds:
	//   nu_t = (C*Delta)^2 * |S|_FD  (explizit -- die implizite Formel des Hauptkernels ist nur
	//   noetig, weil |S| aus Pi selbst von tau abhaengt; |S|_FD tut das nicht),
	//   tau_eff = tau0 + 3*nu_t,  w = 1/tau_eff.
	// stream_collide liest fac_wfd im NAECHSTEN Schritt (ein Schritt Versatz, dt ~ 1e-5 s physikalisch
	// irrelevant) -- getrennter Launch nach stream_collide in derselben In-Order-Queue = deterministisch.
	// Solid-Nachbar: u = 0 (no-slip erster Ordnung), identisch zur g-Diagnose.
	const uint gid = get_global_id(0);
	if(gid>=gd_N) return;
	const uxx n = (uxx)gd_zellen[gid];
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	float g[3][3];
	for(uint a=0u; a<3u; a++) {
		const uxx np=j[2u*a+1u], nm=j[2u*a+2u];
		const bool sp=(flags[np]&TYPE_BO)==TYPE_S, sm=(flags[nm]&TYPE_BO)==TYPE_S;
		for(uint i=0u; i<3u; i++) {
			const float up_ = sp?0.0f:load_u(u, (ulong)i*def_N+(ulong)np);
			const float um_ = sm?0.0f:load_u(u, (ulong)i*def_N+(ulong)nm);
			g[i][a] = 0.5f*(up_-um_);
		}
	}
	float SS=0.0f;
	for(uint i=0u;i<3u;i++) for(uint a=0u;a<3u;a++) { const float Sia=0.5f*(g[i][a]+g[a][i]); SS=fma(Sia,Sia,SS); }
	const float snorm_fd = sqrt(2.0f*SS);
	const float tau0 = 1.0f/def_w;
)+"#ifndef SGS_SISM"+R(
	fac_wfd[gid] = 1.0f/(tau0+3.0f*0.030021f*snorm_fd); // 0.030021 = (C*Delta)^2, C = 0.1733 wie Hauptkernel (0.76421222/(18*sqrt(2)))
)+"#else"+R(
	// ★★ SHEAR-IMPROVED SMAGORINSKY (CFD_SGS_SISM, 07.09.2026, Leveque/Toschi/Shao/Bertoglio JFM 570 (2007)):
	//   nu_t = c2 * max(0, |S| - |<S>|),  c2 = 0.030021 = (C*Delta)^2 wie oben.
	// <S> = EMA der SECHS unabhaengigen S-Komponenten je Facette (fac_sb[6 gid ..]), Start 0, KEIN Warmstart
	// mit S (der lieferte nu_t = 0 im ersten Schritt); Sbar = sqrt(2 <S>:<S>) = Betrag des gemittelten
	// TENSORS -- die billige Form <|S|> waere ein anderes Modell (im Zeitmittel nu_t = 0 = WANDFREI).
	// Zweiphasig (Warmlaufsperre, Muster def_sgs_diag_ab): t < def_sgs_sism_ab -> klassische FDWAND-Formel
	// WORTGLEICH (bei ab >= n_steps bitgleich zum FDWAND-Arm), die EMA laeuft schon mit; danach Abzug mit
	// Klemme fmax(0, .) -- die Klemme ist ZWINGEND, ohne sie faellt tau ab Sbar > 1,57e-4 unter 0,5.
	// Reihenfolge: ALTES Sbar lesen -> w schreiben -> EMA aktualisieren (erster Schritt: Sbar = 0).
	// Racefrei: 1 Work-Item = 1 Facette (Waechter in alloc_facetten_domain), fac_sb[6 gid ..] liest und
	// schreibt nur das eigene Item; u/flags stammen aus dem vorigen In-Order-Launch. Atomics nur an den
	// Zaehlern 126/127 -> physikneutral. alpha = 1/T exakt aus der Schrittzahl (Compile-Konstante).
	const ulong k6 = 6ul*(ulong)gid;
	const float Sn[6] = { g[0][0], g[1][1], g[2][2], 0.5f*(g[0][1]+g[1][0]), 0.5f*(g[0][2]+g[2][0]), 0.5f*(g[1][2]+g[2][1]) };
	float sb[6]; for(uint q=0u; q<6u; q++) sb[q] = fac_sb[k6+(ulong)q];
	const float sbar = sqrt(2.0f*(sq(sb[0])+sq(sb[1])+sq(sb[2])+2.0f*(sq(sb[3])+sq(sb[4])+sq(sb[5]))));
	if(sbar_out!=0u) {
		// ★ 08.09. BANDMODUS: der Kernel liefert NUR Sbar. Kein w-Ersatz -- den Abzug rechnet
		// stream_collide auf dem dort gebildeten Smagorinsky-nu_t (Kipptest-Lehre, s. dort).
		// Vor der Sperre 0 ausgeben: dann ist der Abzug exakt null und der Arm bitgleich zum Bezug.
		fac_wfd[gid] = t<def_sgs_sism_ab ? 0.0f : sbar;
		if(t>=def_sgs_sism_ab&&t%def_zaehl_takt==0ul&&rho_clamp_hits[126]<0xF0000000u) atomic_inc(&rho_clamp_hits[126]);
	} else if(t<def_sgs_sism_ab) {
		fac_wfd[gid] = 1.0f/(tau0+3.0f*0.030021f*snorm_fd); // Phase 1: klassisch, WORTGLEICH zur Zeile im #ifndef-Zweig
	} else {
		const float ds = snorm_fd-sbar;
		fac_wfd[gid] = 1.0f/(tau0+3.0f*0.030021f*fmax(0.0f, ds)); // Phase 2: Abzug mit Klemme
		if(t%def_zaehl_takt==0ul) { // Wirkpfad (t%100 wie ueblich, saettigend): 126 = Abzug aktiv, 127 = Klemme greift (|S| < Sbar)
			if(rho_clamp_hits[126]<0xF0000000u) atomic_inc(&rho_clamp_hits[126]);
			if(ds<0.0f&&rho_clamp_hits[127]<0xF0000000u) atomic_inc(&rho_clamp_hits[127]);
		}
	}
	const float a_ = 1.0f/(float)def_sgs_sism_T;
	for(uint q=0u; q<6u; q++) fac_sb[k6+(ulong)q] = fma(a_, Sn[q]-sb[q], sb[q]); // sb += alpha*(S - sb)
)+"#endif"+R( // SGS_SISM
} // sgs_fdwand()

)+"#ifdef FACETTEN_APG"+R(
float apg_rho_zelle(const uxx nb, const global fpxx* fi, const ulong tt TS_P) { // ★ 16.09.2026 APG: rho EINER Zelle aus den DDFs, Muster rho_rek_ebene (load_f + calculate_rho_u, MIT RHO_CLAMP wie gespeichert). Private Arrays nur mit Konstantindizes (in neighbors/load_f/calculate_rho_u) -- scratch-frei wie dort.
)+"#ifdef FACETTEN_APG_HAKEN3"+R(
	return 1.0f+0.0009765625f*(float)coordinates(nb).x; // ★ TESTHAKEN 3 (16.09.): analytisches rho = 1 + x/1024 -- exakt darstellbar, Achsdifferenzen exakt 2^-10; Host prueft gx bitgenau
)+"#else"+R(
	uxx jj[def_velocity_set]; float fh[def_velocity_set]; float rr, ux_, uy_, uz_;
	neighbors(nb, jj); load_f(nb, fh, fi, jj, tt TS_A); calculate_rho_u(fh, &rr, &ux_, &uy_, &uz_);
	return rr;
)+"#endif"+R( // FACETTEN_APG_HAKEN3
}
)+"#endif"+R( // FACETTEN_APG
)+R(kernel void fac_nachbar_ab(const global velxx* u, const global uchar* flags, const global float* fac_geo,
	const global uint* gd_zellen, const uint gd_N, global float* fac_nb TS_P) { // ★ 16.09.: unveraendert bis auf def_nb_stride -- der APG-Gradient steht im EIGENEN Kernel fac_apg_ab (Gate-Befund: gemeinsam spillte er 1280 B auf der B70)
	// ★★ DETERMINISTISCHE NACHBARABTASTUNG (CFD_FAC_NACHBAR, 03.09.2026). Der Direktzugriff u[nb] in
	// apply_facette_imem lief im Kernel stream_collide, der u im selben Launch schreibt -- gemessen NICHT
	// bitreproduzierbar (xu_det_mit_a/b: cf 0,00073682648 gegen 0,00073630592; ohne NACHBAR bitgleich).
	// Jetzt wie fac_wfd: eigener Launch NACH stream_collide auf dem FERTIGEN u-Feld; die Werte gelten fuer
	// den NAECHSTEN Schritt (ein Schritt Versatz, dt ~ 1e-5 s physikalisch irrelevant). Suche wortgleich
	// zum bisherigen Inline-Block. Ausgabe je Facette (gid == fid, dieselbe F-Reihenfolge wie fac_idx):
	//   fac_nb[2f]   = u_t der zweiten Fluidzelle entlang der Normale (>1e-6 -> angewandt, Slot 72),
	//                  0 = Nachbar gefunden, steht still (Slot 74), -1 = kein Fluidnachbar (Slot 73)
	//   fac_nb[2f+1] = Wandabstand der Abtastzelle (y_w + c_ib . n)
	const uint gid = get_global_id(0);
	if(gid>=gd_N) return;
	const uxx n = (uxx)gd_zellen[gid];
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	const uxx b = 8ul*(uxx)gid;
	const float nx=fac_geo[b], ny=fac_geo[b+1ul], nz=fac_geo[b+2ul], yw=fac_geo[b+3ul];
	// ★ 11.09.2026 SCRATCH-FIX. Frueher wurden nach der Schleife c(ib) und j[ib] mit dem
	// LAUFZEIT-Index ib gelesen. Ein laufzeitindiziertes privates Array wird speicherheimisch:
	// gemessen private_size 7296 auf der B70 und 3648 auf der iGPU, also genau die Fehlerklasse,
	// gegen die scratch_gate.sh gebaut wurde. Gefunden wurde sie erst, als das Gate alle Kernel
	// prueft statt nur stream_collide. Unrolling half nachweislich NICHT -- es ist der Zugriff,
	// nicht die Schleife. Abhilfe: die Werte dort mitnehmen, wo der Index ia compilezeitkonstant
	// ist. Bitgleich, weil es dieselben Werte derselben Iteration sind.
	float bestp = 0.5f; uint ib = 0u; // Schwelle: Link muss ueberwiegend in Normalenrichtung zeigen
	float bcx = 0.0f, bcy = 0.0f, bcz = 0.0f; uxx bnb = (uxx)0ul; // Mitschrift des besten Links
	for(uint ia=1u; ia<def_velocity_set; ia++) {
		if((flags[j[ia]]&TYPE_BO)!=0u) continue; // nur reines Fluid (kein Solid, kein TYPE_E/MS)
		const float cxa=c(ia), cya=c(def_velocity_set+ia), cza=c(2u*def_velocity_set+ia);
		const float cl = sqrt(cxa*cxa+cya*cya+cza*cza);
		const float pr = (cxa*nx+cya*ny+cza*nz)/cl;
		if(pr>bestp) { bestp=pr; ib=ia; bcx=cxa; bcy=cya; bcz=cza; bnb=j[ia]; }
	}
	float utb = -1.0f, ywb = yw;
)+"#ifdef FAC_REK"+R(
	// ★ 23.09. abends, Pruefbefund M1: diese drei standen AUSSERHALB jedes #ifdef und wurden auch
	// bei ausgeschaltetem FAC_REK berechnet und nie gelesen -- toter Code samt einer Division je
	// Facette und Schritt. Damit war der AUS-Arm nicht mehr quelltextidentisch, obwohl ich genau
	// das behauptet hatte.
	float tnx = 0.0f;
	float tny = 0.0f;
	float tnz = 0.0f;
)+"#endif"+R( // FAC_REK
	if(ib>0u) {
		const uxx nb = bnb;
		const float ubx=load_u(u, nb), uby=load_u(u, def_N+(ulong)nb), ubz=load_u(u, 2ul*def_N+(ulong)nb);
		const float undb = nx*ubx+ny*uby+nz*ubz;
		const float utxb=ubx-undb*nx, utyb=uby-undb*ny, utzb=ubz-undb*nz;
		const float ut2 = sqrt(utxb*utxb+utyb*utyb+utzb*utzb);
		utb = (ut2>1e-6f) ? ut2 : 0.0f;
)+"#ifdef FAC_REK"+R(
		const float rinv = (ut2>1e-6f) ? 1.0f/ut2 : 0.0f;
		tnx = utxb*rinv;
		tny = utyb*rinv;
		tnz = utzb*rinv;
)+"#endif"+R( // FAC_REK
		ywb = yw + (bcx*nx+bcy*ny+bcz*nz); // war c(ib)... -- siehe Scratch-Fix oben
	}
	fac_nb[def_nb_stride*(ulong)gid] = utb;
	fac_nb[def_nb_stride*(ulong)gid+1ul] = ywb;
)+"#ifdef FAC_REK"+R(
	fac_nb[def_nb_stride*(ulong)gid+def_nb_roff+0ul] = tnx;
	fac_nb[def_nb_stride*(ulong)gid+def_nb_roff+1ul] = tny;
	fac_nb[def_nb_stride*(ulong)gid+def_nb_roff+2ul] = tnz;
)+"#endif"+R(
 // def_nb_stride = 2 (5 unter FACETTEN_APG), JIT-Emission lbm.cpp
} // fac_nachbar_ab()
)+"#ifdef FACETTEN_APG"+R(
kernel void fac_apg_ab(const global uchar* flags, const global uint* gd_zellen, const uint gd_N, global float* fac_nb,
	const global fpxx* fi, const ulong t, global uint* rho_clamp_hits TS_P) { // ★ 16.09.2026 APG-VORKERNEL (eigener Kernel: Gate-Befund, im gemeinsamen Kernel spillte der Gradient 1280 B auf der B70)
	const uint gid = get_global_id(0);
	if(gid>=gd_N) return;
	const uxx n = (uxx)gd_zellen[gid];
	uxx j[def_velocity_set]; neighbors(n, j);
	{ // ★ 16.09.2026 APG, DETERMINISTISCHES NACHBAR-RHO (PLAN-APG-2026-09-16.md §B, billigste Form nach Heikos Vorgabe):
	  // grad(rho) an der Facettenzelle aus den DDFs der SECHS ACHSNACHBARN -- zentrale Differenz; einseitig, wo ein Nachbar
	  // Solid ist; 0 auf einer Achse, auf der beide fehlen ([307]). KEIN rho-Puffer wird gelesen: RHO_RAND, RHO_SPARSAM und
	  // RHO_FP16 bleiben unberuehrt, die alten Sperren entfallen. rho je Zelle wie in rho_rek_ebene: load_f(t+1) +
	  // calculate_rho_u = das rho, das stream_collide(t+1) dort selbst bilden wird (MIT RHO_CLAMP, wie gespeichert) --
	  // synchron zum rhon der Facette im naechsten Schritt statt des alten t-1/t-Gemischs (Befund A3, bitreproduzierbar).
	  // TYPE_S wird uebersprungen (kein rho); TYPE_E/TYPE_MS tragen gueltiges rho (Befund A4). Verkehr: 7 Zellen x
	  // (19 x 2 B + 1 B) je Facette und Schritt = 273 B/Facette (Rechnung; Cache-Wiederverwendung ungemessen).
	  // Achsen-Erkennung ueber c(ia), NICHT ueber die Indexordnung: ein Achslink hat |cx|+|cy|+|cz| = 1. Beitrag eines
	  // Nachbarn e mit Dichte rho_e zur Ableitung entlang +Achse: (rho_e - rho_0) * (e . Achse); beide vorhanden ->
	  // Mittel = zentrale Differenz (rho_+ - rho_-)/2, einer -> einseitig. Keine laufzeitindizierten privaten Arrays
	  // ausser j[]/jj[] in der ia-Schleife (Muster der Schleife oben, Scratch-Gate-belegt).
	  // SCRATCH-LEHRE (11.09., Gate-Befund 16.09. private=7296): eine ia-Schleife mit Funktionsaufrufen (neighbors/load_f) wird
	  // NICHT ausgerollt, j[ia] wird zum Laufzeitindex und das private Array speicherheimisch. Darum SECHS ausgeschriebene
	  // Bloecke mit LITERALEN Linkindizes 1..6 -- die Achsenerkennung ueber c(ia) faltet der Compiler bei Literalen weg.
	  const float r0 = apg_rho_zelle(n, fi, t+1ul TS_A);
	  float sx=0.0f, sy=0.0f, sz=0.0f, kx=0.0f, ky=0.0f, kz=0.0f; // Summen und Nachbarzahl je Achse
	  if(fabs(c(1u))+fabs(c(def_velocity_set+1u))+fabs(c(2u*def_velocity_set+1u))==1.0f&&(flags[j[1]]&TYPE_BO)!=TYPE_S) { const float d_=apg_rho_zelle(j[1], fi, t+1ul TS_A)-r0; sx=fma(d_,c(1u),sx); sy=fma(d_,c(def_velocity_set+1u),sy); sz=fma(d_,c(2u*def_velocity_set+1u),sz); kx+=fabs(c(1u)); ky+=fabs(c(def_velocity_set+1u)); kz+=fabs(c(2u*def_velocity_set+1u)); }
	  if(fabs(c(2u))+fabs(c(def_velocity_set+2u))+fabs(c(2u*def_velocity_set+2u))==1.0f&&(flags[j[2]]&TYPE_BO)!=TYPE_S) { const float d_=apg_rho_zelle(j[2], fi, t+1ul TS_A)-r0; sx=fma(d_,c(2u),sx); sy=fma(d_,c(def_velocity_set+2u),sy); sz=fma(d_,c(2u*def_velocity_set+2u),sz); kx+=fabs(c(2u)); ky+=fabs(c(def_velocity_set+2u)); kz+=fabs(c(2u*def_velocity_set+2u)); }
	  if(fabs(c(3u))+fabs(c(def_velocity_set+3u))+fabs(c(2u*def_velocity_set+3u))==1.0f&&(flags[j[3]]&TYPE_BO)!=TYPE_S) { const float d_=apg_rho_zelle(j[3], fi, t+1ul TS_A)-r0; sx=fma(d_,c(3u),sx); sy=fma(d_,c(def_velocity_set+3u),sy); sz=fma(d_,c(2u*def_velocity_set+3u),sz); kx+=fabs(c(3u)); ky+=fabs(c(def_velocity_set+3u)); kz+=fabs(c(2u*def_velocity_set+3u)); }
	  if(fabs(c(4u))+fabs(c(def_velocity_set+4u))+fabs(c(2u*def_velocity_set+4u))==1.0f&&(flags[j[4]]&TYPE_BO)!=TYPE_S) { const float d_=apg_rho_zelle(j[4], fi, t+1ul TS_A)-r0; sx=fma(d_,c(4u),sx); sy=fma(d_,c(def_velocity_set+4u),sy); sz=fma(d_,c(2u*def_velocity_set+4u),sz); kx+=fabs(c(4u)); ky+=fabs(c(def_velocity_set+4u)); kz+=fabs(c(2u*def_velocity_set+4u)); }
	  if(fabs(c(5u))+fabs(c(def_velocity_set+5u))+fabs(c(2u*def_velocity_set+5u))==1.0f&&(flags[j[5]]&TYPE_BO)!=TYPE_S) { const float d_=apg_rho_zelle(j[5], fi, t+1ul TS_A)-r0; sx=fma(d_,c(5u),sx); sy=fma(d_,c(def_velocity_set+5u),sy); sz=fma(d_,c(2u*def_velocity_set+5u),sz); kx+=fabs(c(5u)); ky+=fabs(c(def_velocity_set+5u)); kz+=fabs(c(2u*def_velocity_set+5u)); }
	  if(fabs(c(6u))+fabs(c(def_velocity_set+6u))+fabs(c(2u*def_velocity_set+6u))==1.0f&&(flags[j[6]]&TYPE_BO)!=TYPE_S) { const float d_=apg_rho_zelle(j[6], fi, t+1ul TS_A)-r0; sx=fma(d_,c(6u),sx); sy=fma(d_,c(def_velocity_set+6u),sy); sz=fma(d_,c(2u*def_velocity_set+6u),sz); kx+=fabs(c(6u)); ky+=fabs(c(def_velocity_set+6u)); kz+=fabs(c(2u*def_velocity_set+6u)); }
	  float gx = (kx>0.0f) ? sx/kx : 0.0f, gy = (ky>0.0f) ? sy/ky : 0.0f, gz = (kz>0.0f) ? sz/kz : 0.0f;
	  if(t%def_zaehl_takt==0ul) { // Zaehler nur an Zaehlschritten (Absturzlehre 15.09.: nichts Atomares im Immerpfad)
		if(rho_clamp_hits[306]<0xF0000000u) atomic_inc(&rho_clamp_hits[306]); // Vorkernel-Besuche (Soll = [7])
		if((kx==0.0f||ky==0.0f||kz==0.0f)&&rho_clamp_hits[307]<0xF0000000u) atomic_inc(&rho_clamp_hits[307]); // entartet: eine Achse ohne Fluidnachbarn
	  }
)+"#ifdef FACETTEN_APG_HAKEN"+R(
	  gx = 1.0e-3f; gy = 0.0f; gz = 0.0f; // ★ TESTHAKEN CFD_FAC_APG_HAKEN=2: Konstantgradient -> Host prueft fac_nb bitgenau und den Durchstich bis dp/ds
)+"#endif"+R( // FACETTEN_APG_HAKEN
)+"#ifdef FACETTEN_APG_HAKEN3"+R(
	  gz = kx; // ★ TESTHAKEN 3: Seitenkanal -- der Host prueft gx == 2^-10 exakt wo kx>0 (sonst 0) und gy == 0
)+"#endif"+R( // FACETTEN_APG_HAKEN3
	  fac_nb[def_nb_stride*(ulong)gid+2ul] = gx; fac_nb[def_nb_stride*(ulong)gid+3ul] = gy; fac_nb[def_nb_stride*(ulong)gid+4ul] = gz;
	}
} // fac_apg_ab()
)+"#endif"+R( // FACETTEN_APG

)+R(kernel void object_torque(const global float* F, const global uint* f_maske, const global uchar* flags, const uchar flag_marker, const float cx, const float cy, const float cz, volatile global float* object_sum) {
	const uxx n = get_global_id(0); // n = x+(y+z*Ny)*Nx
	const uint lid = get_local_id(0); // local memory reduction of cl_workgroup_size:1
	local float3 cache[cl_workgroup_size];
	// ★ KORREKTUR 2026-08-08 (Pruefer-Befund): hier stand load3(F, n) mit dem VOLL-Domaenen-Index.
	// F ist im Fork nur so gross wie die Bounding-Box um den Koerper (3*def_FBN statt 3*def_N), also
	// las das weit hinter dem Puffer. Latent, weil object_torque von keinem Setup gerufen wird -- aber
	// es ist eine Falle fuer den Tag, an dem jemand Momente auswertet. Alle uebrigen F-Zugriffe des
	// Forks waren bereits auf load3_F umgestellt, nur dieser eine nicht.
	cache[lid] = n<(uxx)def_N&&flags[n]==flag_marker ? cross(position(coordinates(n))-(float3)(cx, cy, cz), load3_F(F, f_maske, n)) : (float3)(0.0f, 0.0f, 0.0f);
	barrier(CLK_GLOBAL_MEM_FENCE);
	for(uint s=1u; s<cl_workgroup_size; s*=2u) {
		if(lid%(2u*s)==0u) cache[lid] += cache[lid+s];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(lid==0u) { // global memory reduction with atomic addition of local_sum
		const float3 local_sum = cache[0];
		if(local_sum.x!=0.0f) atomic_add_f(&object_sum[0], local_sum.x);
		if(local_sum.y!=0.0f) atomic_add_f(&object_sum[1], local_sum.y);
		if(local_sum.z!=0.0f) atomic_add_f(&object_sum[2], local_sum.z);
	}
} // object_torque()
)+"#endif"+R( // FORCE_FIELD

)+"#ifdef PARTICLES"+R(
)+"#ifdef FORCE_FIELD"+R(
// ★ Audit-Befund 19 (latent, PARTICLES ist wegkompiliert): spread_force indiziert F mit dem
// VOLLEN Gitterindex, nicht mit der F-Bounding-Box. Wer PARTICLES je einschaltet, muss hier
// zuerst auf FBI-Indexierung umstellen (Muster: update_force_field), sonst schreibt es daneben.
)+R(void spread_force(volatile global float* F, const float3 p, const float3 Fn) {
	const float xa=p.x-0.5f+1.5f*(float)def_Nx, ya=p.y-0.5f+1.5f*(float)def_Ny, za=p.z-0.5f+1.5f*(float)def_Nz; // subtract lattice offsets
	const uint xb=(uint)xa, yb=(uint)ya, zb=(uint)za; // integer casting to find bottom left corner
	const float x1=xa-(float)xb, y1=ya-(float)yb, z1=za-(float)zb; // calculate interpolation factors
	for(uint c=0u; c<8u; c++) { // count over eight corner points
		const uint i=(c&0x04u)>>2, j=(c&0x02u)>>1, k=c&0x01u; // disassemble c into corner indices ijk
		const uint x=(xb+i)%def_Nx, y=(yb+j)%def_Ny, z=(zb+k)%def_Nz; // calculate corner lattice positions
		const uxx n = (uxx)x+(uxx)(y+z*def_Ny)*(uxx)def_Nx; // calculate lattice linear index
		const float d = (1.0f-fabs(x1-(float)i))*(1.0f-fabs(y1-(float)j))*(1.0f-fabs(z1-(float)k)); // force spreading
		const float3 Fnd = Fn*d;
		if(Fnd.x!=0.0f) atomic_add_f(&F[                 n], Fnd.x); // F[                 n] += Fnd.x;
		if(Fnd.y!=0.0f) atomic_add_f(&F[    def_N+(ulong)n], Fnd.y); // F[    def_N+(ulong)n] += Fnd.y;
		if(Fnd.z!=0.0f) atomic_add_f(&F[2ul*def_N+(ulong)n], Fnd.z); // F[2ul*def_N+(ulong)n] += Fnd.z;
	}
} // spread_force()
)+"#endif"+R( // FORCE_FIELD

)+R(float3 particle_boundary_force(const float3 p, const global uchar* flags) { // normalized pseudo-force to prevent particles from entering solid boundaries or exiting fluid phase
	const float xa=p.x-0.5f+1.5f*(float)def_Nx, ya=p.y-0.5f+1.5f*(float)def_Ny, za=p.z-0.5f+1.5f*(float)def_Nz; // subtract lattice offsets
	const uint xb=(uint)xa, yb=(uint)ya, zb=(uint)za; // integer casting to find bottom left corner
	const float x1=xa-(float)xb, y1=ya-(float)yb, z1=za-(float)zb; // calculate interpolation factors
	float3 boundary_force = (float3)(0.0f, 0.0f, 0.0f);
	float boundary_distance = 2.0f;
	for(uint c=0u; c<8u; c++) { // count over eight corner points
		const uint i=(c&0x04u)>>2, j=(c&0x02u)>>1, k=c&0x01u; // disassemble c into corner indices ijk
		const uint x=(xb+i)%def_Nx, y=(yb+j)%def_Ny, z=(zb+k)%def_Nz; // calculate corner lattice positions
		const uxx n = (uxx)x+(uxx)(y+z*def_Ny)*(uxx)def_Nx; // calculate lattice linear index
		if(flags[n]&(TYPE_S|TYPE_G)) {
			boundary_force += (float3)(0.5f, 0.5f, 0.5f)-(float3)((float)i, (float)j, (float)k);
			boundary_distance = fmin(boundary_distance, length((float3)(x1, y1, z1)-(float3)((float)i, (float)j, (float)k)));
		}
	}
	const float particle_radius = 0.5f; // has to be between 0.0f and 0.5f, default: 0.5f (hydrodynamic radius)
	return boundary_distance-0.5f<particle_radius ? normalize(boundary_force) : (float3)(0.0f, 0.0f, 0.0f);
} // particle_boundary_force()
)+R(bool position_is_in_domain_including_halo(const float3 p) {
	const float hNx = 0.5f*(float)(def_Nx-(def_Dx>1u)); // subtract half of halo still
	const float hNy = 0.5f*(float)(def_Ny-(def_Dy>1u));
	const float hNz = 0.5f*(float)(def_Nz-(def_Dz>1u));
	return p.x>=-hNx&&p.x<hNx&&p.y>=-hNy&&p.y<hNy&&p.z>=-hNz&&p.z<hNz;
}
)+R(bool position_is_in_domain_excluding_halo(const float3 p) {
	const float hNx = 0.5f*(float)(def_Nx-2u*(def_Dx>1u)); // subtract full halo
	const float hNy = 0.5f*(float)(def_Ny-2u*(def_Dy>1u));
	const float hNz = 0.5f*(float)(def_Nz-2u*(def_Dz>1u));
	return p.x>=-hNx&&p.x<hNx&&p.y>=-hNy&&p.y<hNy&&p.z>=-hNz&&p.z<hNz;
}
)+R(kernel void integrate_particles)+"("+R(global float* particles, const global float* u, const global uchar* flags, const float time_step_multiplicator // ) {
)+"#ifdef FORCE_FIELD"+R(
	, volatile global float* F, const float fx, const float fy, const float fz
)+"#endif"+R( // FORCE_FIELD
)+") {"+R( // integrate_particles()
	const uxx n = get_global_id(0); // index of membrane points
	if(n>=(uxx)def_particles_N) return;
	float3 p = (float3)(particles[n], particles[def_particles_N+(ulong)n], particles[2ul*def_particles_N+(ulong)n]); // cache particle position
	p = mirror_position(p); // mirror into global simulation box
	p -= (float3)(def_domain_offset_x, def_domain_offset_y, def_domain_offset_z); // subtract domain offset, then treat point in local domain
	if(def_Dx*def_Dy*def_Dz>1u&&!position_is_in_domain_including_halo(p)) {
		p.x = as_float(0xFFFFFFFFu); // invalidate x-coordinate for all particles outside of the local domain (including halo)
	} else {
)+"#ifdef FORCE_FIELD"+R(
		if(def_particles_rho!=1.0f) { // apply volume force for all particles in local domain (including halo)
			const float drho = def_particles_rho-1.0f; // density difference leads to particle buoyancy
			float3 Fn = (float3)(fx*drho, fy*drho, fz*drho); // F = F_p+F_f = (m_p-m_f)*g = (rho_p-rho_f)*g*V
			spread_force(F, p, Fn); // do force spreading
		}
)+"#endif"+R( // FORCE_FIELD
		if(def_Dx*def_Dy*def_Dz>1u&&!position_is_in_domain_excluding_halo(p)) { // skip remaining ghost particles in halo
			p.x = as_float(0xFFFFFFFFu); // invalidate x-coordinate for all particles outside of the local domain
		} else { // advect only particles in local domain (excluding halo)
			float3 un = interpolate_u(u, p); // trilinear interpolation of velocity at point p
			un = (un+length(un)*particle_boundary_force(p, flags))*time_step_multiplicator;
			p += un; // advect particles
			p += (float3)(def_domain_offset_x, def_domain_offset_y, def_domain_offset_z); // add domain offset, back to global domain
			p = mirror_position(p); // mirror advected position again into global simulation box
			particles[    def_particles_N+(ulong)n] = p.y; // store y/z-coordinates only for particles in domain
			particles[2ul*def_particles_N+(ulong)n] = p.z;
		}
	}
	particles[n] = p.x; // always store x-coordinate (invalidated or particles in domain)
} // integrate_particles()
)+"#endif"+R( // PARTICLES



)+R(uint get_area(const uint direction) {
	const uint A[3] = { def_Ax, def_Ay, def_Az };
	return A[direction];
}
)+R(uxx index_extract_p(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(def_Nx-2u, a%def_Ny, a/def_Ny), (uint3)(a/def_Nz, def_Ny-2u, a%def_Nz), (uint3)(a%def_Nx, a/def_Nx, def_Nz-2u) };
	return index(coordinates[direction]);
}
)+R(uxx index_extract_m(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(       1u, a%def_Ny, a/def_Ny), (uint3)(a/def_Nz,        1u, a%def_Nz), (uint3)(a%def_Nx, a/def_Nx,        1u) };
	return index(coordinates[direction]);
}
)+R(uxx index_insert_p(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(def_Nx-1u, a%def_Ny, a/def_Ny), (uint3)(a/def_Nz, def_Ny-1u, a%def_Nz), (uint3)(a%def_Nx, a/def_Nx, def_Nz-1u) };
	return index(coordinates[direction]);
}
)+R(uxx index_insert_m(const uint a, const uint direction) {
	const uint3 coordinates[3] = { (uint3)(       0u, a%def_Ny, a/def_Ny), (uint3)(a/def_Nz,        0u, a%def_Nz), (uint3)(a%def_Nx, a/def_Nx,        0u) };
	return index(coordinates[direction]);
}

)+R(uint index_transfer(const uint side_i) {
	const uchar index_transfer_data[2u*def_dimensions*def_transfers] = {
)+"#if defined(D2Q9)"+R(
		1,  5,  7, // xp
		2,  6,  8, // xm
		3,  5,  8, // yp
		4,  6,  7  // ym
)+"#elif defined(D3Q15)"+R(
		1,  7, 14,  9, 11, // xp
		2,  8, 13, 10, 12, // xm
		3,  7, 12,  9, 13, // yp
		4,  8, 11, 10, 14, // ym
		5,  7, 10, 11, 13, // zp
		6,  8,  9, 12, 14  // zm
)+"#elif defined(D3Q19)"+R(
		1,  7, 13,  9, 15, // xp
		2,  8, 14, 10, 16, // xm
		3,  7, 14, 11, 17, // yp
		4,  8, 13, 12, 18, // ym
		5,  9, 16, 11, 18, // zp
		6, 10, 15, 12, 17  // zm
)+"#elif defined(D3Q27)"+R(
		1,  7, 13,  9, 15, 19, 26, 21, 23, // xp
		2,  8, 14, 10, 16, 20, 25, 22, 24, // xm
		3,  7, 14, 11, 17, 19, 24, 21, 25, // yp
		4,  8, 13, 12, 18, 20, 23, 22, 26, // ym
		5,  9, 16, 11, 18, 19, 22, 23, 25, // zp
		6, 10, 15, 12, 17, 20, 21, 24, 26  // zm
)+"#endif"+R( // D3Q27
	};
	return (uint)index_transfer_data[side_i];
}
)+R(void extract_fi(const uint a, const uint A, const uxx n, const uint side, const ulong t, global fpxx_copy* transfer_buffer, const global fpxx_copy* fi TS_P) { // FORK: TS_P, sonst kennt index_f hier kein tile_slot
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	for(uint b=0u; b<def_transfers; b++) {
		const uint i = index_transfer(side*def_transfers+b);
		const ulong index = index_f(i%2u ? j[i] : n, t%2ul ? (i%2u ? i+1u : i-1u) : i); // Esoteric-Pull: standard store, or streaming part 1/2
		transfer_buffer[b*A+a] = fi[index]; // fpxx_copy allows direct copying without decompression+compression
	}
}
)+R(void insert_fi(const uint a, const uint A, const uxx n, const uint side, const ulong t, const global fpxx_copy* transfer_buffer, global fpxx_copy* fi TS_P) { // FORK: TS_P
	uxx j[def_velocity_set]; // neighbor indices
	neighbors(n, j); // calculate neighbor indices
	for(uint b=0u; b<def_transfers; b++) {
		const uint i = index_transfer(side*def_transfers+b);
		const ulong index = index_f(i%2u ? n : j[i-1u], t%2ul ? i : (i%2u ? i+1u : i-1u)); // Esoteric-Pull: standard load, or streaming part 2/2
		fi[index] = transfer_buffer[b*A+a]; // fpxx_copy allows direct copying without decompression+compression
	}
}
)+R(kernel void transfer_extract_fi(const uint direction, const ulong t, global fpxx_copy* transfer_buffer_p, global fpxx_copy* transfer_buffer_m, const global fpxx_copy* fi TS_P) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_fi(a, A, index_extract_p(a, direction), 2u*direction+0u, t, transfer_buffer_p, fi TS_A);
	extract_fi(a, A, index_extract_m(a, direction), 2u*direction+1u, t, transfer_buffer_m, fi TS_A);
}
)+R(kernel void transfer__insert_fi(const uint direction, const ulong t, const global fpxx_copy* transfer_buffer_p, const global fpxx_copy* transfer_buffer_m, global fpxx_copy* fi TS_P) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_fi(a, A, index_insert_p(a, direction), 2u*direction+0u, t, transfer_buffer_p, fi TS_A);
	insert_fi(a, A, index_insert_m(a, direction), 2u*direction+1u, t, transfer_buffer_m, fi TS_A);
}

)+R(void extract_rho_u_flags(const uint a, const uint A, const uxx n, global char* transfer_buffer, const global rhoxx* rho, const global velxx* u, const global uchar* flags) {
	((global float*)transfer_buffer)[      a] = load_rho(rho,      n); // Puffer bleibt float32: Stride 17 unveraendert
	((global float*)transfer_buffer)[    A+a] = load_u(u, n);
	((global float*)transfer_buffer)[ 2u*A+a] = load_u(u, def_N+(ulong)n);
	((global float*)transfer_buffer)[ 3u*A+a] = load_u(u, 2ul*def_N+(ulong)n);
	((global uchar*)transfer_buffer)[16u*A+a] = flags[             n];
}
)+R(void insert_rho_u_flags(const uint a, const uint A, const uxx n, const global char* transfer_buffer, global rhoxx* rho, global velxx* u, global uchar* flags) {
	store_rho(rho,     n,   ((const global float*)transfer_buffer)[      a]);
	store_u(u, n, ((const global float*)transfer_buffer)[    A+a]);
	store_u(u, def_N+(ulong)n, ((const global float*)transfer_buffer)[ 2u*A+a]);
	store_u(u, 2ul*def_N+(ulong)n, ((const global float*)transfer_buffer)[ 3u*A+a]);
	flags[             n] = ((const global uchar*)transfer_buffer)[16u*A+a];
}
)+R(kernel void transfer_extract_rho_u_flags(const uint direction, const ulong t, global char* transfer_buffer_p, global char* transfer_buffer_m, const global rhoxx* rho, const global velxx* u, const global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_rho_u_flags(a, A, index_extract_p(a, direction), transfer_buffer_p, rho, u, flags);
	extract_rho_u_flags(a, A, index_extract_m(a, direction), transfer_buffer_m, rho, u, flags);
}
)+R(kernel void transfer__insert_rho_u_flags(const uint direction, const ulong t, const global char* transfer_buffer_p, const global char* transfer_buffer_m, global rhoxx* rho, global velxx* u, global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_rho_u_flags(a, A, index_insert_p(a, direction), transfer_buffer_p, rho, u, flags);
	insert_rho_u_flags(a, A, index_insert_m(a, direction), transfer_buffer_m, rho, u, flags);
}

)+R(kernel void transfer_extract_flags(const uint direction, const ulong t, global uchar* transfer_buffer_p, global uchar* transfer_buffer_m, const global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	transfer_buffer_p[a] = flags[index_extract_p(a, direction)];
	transfer_buffer_m[a] = flags[index_extract_m(a, direction)];
}
)+R(kernel void transfer__insert_flags(const uint direction, const ulong t, const global uchar* transfer_buffer_p, const global uchar* transfer_buffer_m, global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	flags[index_insert_p(a, direction)] = transfer_buffer_p[a];
	flags[index_insert_m(a, direction)] = transfer_buffer_m[a];
}

)+"#ifdef FORCE_FIELD"+R(
)+R(void extract_F(const uint a, const uint A, const uxx n, global float* transfer_buffer, const global float* F) {
	transfer_buffer[     a] = F[                 n];
	transfer_buffer[   A+a] = F[    def_N+(ulong)n];
	transfer_buffer[2u*A+a] = F[2ul*def_N+(ulong)n];
}
)+R(void insert_F(const uint a, const uint A, const uxx n, const global float* transfer_buffer, global float* F) {
	F[                 n] = transfer_buffer[     a];
	F[    def_N+(ulong)n] = transfer_buffer[   A+a];
	F[2ul*def_N+(ulong)n] = transfer_buffer[2u*A+a];
}
// ★ Gross-Audit LATENT->FIX: transfer_extract_F/insert_F indizieren F voll-domaenig -- mit F-BBox
// (F nur 3*def_FBN gross) waere das OOB. Multi-GPU+BBox ist deshalb host-seitig hart verweigert
// (lbm.cpp, Konstruktor-Guard); wer die Kombination baut, muss hier auf f_bbox() umstellen.
)+R(kernel void transfer_extract_F(const uint direction, const ulong t, global float* transfer_buffer_p, global float* transfer_buffer_m, const global float* F) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_F(a, A, index_extract_p(a, direction), transfer_buffer_p, F);
	extract_F(a, A, index_extract_m(a, direction), transfer_buffer_m, F);
}
)+R(kernel void transfer__insert_F(const uint direction, const ulong t, const global float* transfer_buffer_p, const global float* transfer_buffer_m, global float* F) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_F(a, A, index_insert_p(a, direction), transfer_buffer_p, F);
	insert_F(a, A, index_insert_m(a, direction), transfer_buffer_m, F);
}
)+"#endif"+R( // FORCE_FIELD

)+"#ifdef SURFACE"+R(
)+R(void extract_phi_massex_flags(const uint a, const uint A, const uxx n, global char* transfer_buffer, const global float* phi, const global float* massex, const global uchar* flags) {
	((global float*)transfer_buffer)[     a] = phi   [n];
	((global float*)transfer_buffer)[   A+a] = massex[n];
	((global uchar*)transfer_buffer)[8u*A+a] = flags [n];
}
)+R(void insert_phi_massex_flags(const uint a, const uint A, const uxx n, const global char* transfer_buffer, global float* phi, global float* massex, global uchar* flags) {
	phi   [n] = ((global float*)transfer_buffer)[     a];
	massex[n] = ((global float*)transfer_buffer)[   A+a];
	flags [n] = ((global uchar*)transfer_buffer)[8u*A+a];
}
)+R(kernel void transfer_extract_phi_massex_flags(const uint direction, const ulong t, global char* transfer_buffer_p, global char* transfer_buffer_m, const global float* phi, const global float* massex, const global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_phi_massex_flags(a, A, index_extract_p(a, direction), transfer_buffer_p, phi, massex, flags);
	extract_phi_massex_flags(a, A, index_extract_m(a, direction), transfer_buffer_m, phi, massex, flags);
}
)+R(kernel void transfer__insert_phi_massex_flags(const uint direction, const ulong t, const global char* transfer_buffer_p, const global char* transfer_buffer_m, global float* phi, global float* massex, global uchar* flags) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_phi_massex_flags(a, A, index_insert_p(a, direction), transfer_buffer_p, phi, massex, flags);
	insert_phi_massex_flags(a, A, index_insert_m(a, direction), transfer_buffer_m, phi, massex, flags);
}
)+"#endif"+R( // SURFACE

)+"#ifdef TEMPERATURE"+R(
)+R(void extract_gi(const uint a, const uxx n, const uint side, const ulong t, global fpxx_copy* transfer_buffer, const global fpxx_copy* gi) {
	uxx j7[7u]; // neighbor indices
	neighbors_temperature(n, j7); // calculate neighbor indices
	const uint i = side+1u;
	const ulong index = index_f(i%2u ? j7[i] : n, t%2ul ? (i%2u ? i+1u : i-1u) : i); // Esoteric-Pull: standard store, or streaming part 1/2
	transfer_buffer[a] = gi[index]; // fpxx_copy allows direct copying without decompression+compression
}
)+R(void insert_gi(const uint a, const uxx n, const uint side, const ulong t, const global fpxx_copy* transfer_buffer, global fpxx_copy* gi) {
	uxx j7[7u]; // neighbor indices
	neighbors_temperature(n, j7); // calculate neighbor indices
	const uint i = side+1u;
	const ulong index = index_f(i%2u ? n : j7[i-1u], t%2ul ? i : (i%2u ? i+1u : i-1u)); // Esoteric-Pull: standard load, or streaming part 2/2
	gi[index] = transfer_buffer[a]; // fpxx_copy allows direct copying without decompression+compression
}
)+R(kernel void transfer_extract_gi(const uint direction, const ulong t, global fpxx_copy* transfer_buffer_p, global fpxx_copy* transfer_buffer_m, const global fpxx_copy* gi) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	extract_gi(a, index_extract_p(a, direction), 2u*direction+0u, t, transfer_buffer_p, gi);
	extract_gi(a, index_extract_m(a, direction), 2u*direction+1u, t, transfer_buffer_m, gi);
}
)+R(kernel void transfer__insert_gi(const uint direction, const ulong t, const global fpxx_copy* transfer_buffer_p, const global fpxx_copy* transfer_buffer_m, global fpxx_copy* gi) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	insert_gi(a, index_insert_p(a, direction), 2u*direction+0u, t, transfer_buffer_p, gi);
	insert_gi(a, index_insert_m(a, direction), 2u*direction+1u, t, transfer_buffer_m, gi);
}

)+R(kernel void transfer_extract_T(const uint direction, const ulong t, global float* transfer_buffer_p, global float* transfer_buffer_m, const global float* T) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	transfer_buffer_p[a] = T[index_extract_p(a, direction)];
	transfer_buffer_m[a] = T[index_extract_m(a, direction)];
}
)+R(kernel void transfer__insert_T(const uint direction, const ulong t, const global float* transfer_buffer_p, const global float* transfer_buffer_m, global float* T) {
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	T[index_insert_p(a, direction)] = transfer_buffer_p[a];
	T[index_insert_m(a, direction)] = transfer_buffer_m[a];
}
)+"#endif"+R( // TEMPERATURE



)+R(kernel void voxelize_mesh)+"("+R(const uint direction, global fpxx* fi, global velxx* u, global uchar* flags, const ulong t, const uchar flag, const global float* p0, const global float* p1, const global float* p2, const global float* bbu // ) { // voxelize triangle mesh
)+"#ifdef SURFACE"+R(
	, global float* mass, global float* massex // argument order is important
)+"#endif"+R( // SURFACE
)+R( TS_P
)+") {"+R( // voxelize_mesh()
	const uint a=get_global_id(0), A=get_area(direction); // a = domain area index for each side, A = area of the domain boundary
	if(a>=A) return; // area might not be a multiple of cl_workgroup_size, so return here to avoid writing in unallocated memory space
	const uint triangle_number = as_uint(bbu[0]);
	const float x0=bbu[ 1], y0=bbu[ 2], z0=bbu[ 3], x1=bbu[ 4], y1=bbu[ 5], z1=bbu[ 6];
	const float cx=bbu[ 7], cy=bbu[ 8], cz=bbu[ 9], ux=bbu[10], uy=bbu[11], uz=bbu[12], rx=bbu[13], ry=bbu[14], rz=bbu[15];
	const uint hmin = direction==0u ? (uint)clamp((int)x0-def_Ox, 0, (int)def_Nx-1) :
	                  direction==1u ? (uint)clamp((int)y0-def_Oy, 0, (int)def_Ny-1) :
	                                  (uint)clamp((int)z0-def_Oz, 0, (int)def_Nz-1);
	const uint hmax = direction==0u ? (uint)clamp((int)x1-def_Ox, 0, (int)def_Nx-1) :
	                  direction==1u ? (uint)clamp((int)y1-def_Oy, 0, (int)def_Ny-1) :
	                                  (uint)clamp((int)z1-def_Oz, 0, (int)def_Nz-1);
	const uint3 xyz = direction==0u ? (uint3)(hmin, a%def_Ny, a/def_Ny) :
	                  direction==1u ? (uint3)(a/def_Nz, hmin, a%def_Nz) :
	                                  (uint3)(a%def_Nx, a/def_Nx, hmin);
	const float3 offset = (float3)(0.5f*(float)((int)def_Nx+2*def_Ox)-0.5f, 0.5f*(float)((int)def_Ny+2*def_Oy)-0.5f, 0.5f*(float)((int)def_Nz+2*def_Oz)-0.5f);
	const float3 r_origin = position(xyz)+offset;
	const float3 r_direction = (float3)((float)(direction==0u), (float)(direction==1u), (float)(direction==2u));
	uint intersections=0u, intersections_check=0u;
	ushort distances[64]; // allow up to 64 mesh intersections
	const bool condition = direction==0u ? r_origin.y<y0||r_origin.z<z0||r_origin.y>=y1||r_origin.z>=z1 : direction==1u ? r_origin.x<x0||r_origin.z<z0||r_origin.x>=x1||r_origin.z>=z1 : r_origin.x<x0||r_origin.y<y0||r_origin.x>=x1||r_origin.y>=y1;
	if(condition) return; // don't use local memory (~25% slower, but this also runs on old OpenCL 1.0 GPUs)
	for(uint i=0u; i<triangle_number; i++) {
		const uint tx=3u*i, ty=tx+1u, tz=ty+1u;
		const float3 p0i = (float3)(p0[tx], p0[ty], p0[tz]);
		const float3 p1i = (float3)(p1[tx], p1[ty], p1[tz]);
		const float3 p2i = (float3)(p2[tx], p2[ty], p2[tz]);
		const float3 u=p1i-p0i, v=p2i-p0i, w=r_origin-p0i, h=cross(r_direction, v), q=cross(w, u); // bidirectional ray-triangle intersection (Moeller-Trumbore algorithm)
		const float g=dot(u, h), f=1.0f/g, s=f*dot(w, h), t=f*dot(r_direction, q), d=f*dot(v, q); // check for division by zero in case g==0, otherwise f=NaN can cause hang
		if(g!=0.0f&&s>=0.0f&&s<1.0f&&t>=0.0f&&s+t<1.0f) { // ray-triangle intersection ahead or behind
			if(d>0.0f) { // ray-triangle intersection ahead
				if(intersections<64u&&d<65536.0f) distances[intersections] = (ushort)d; // store distance to intersection in array as ushort
				intersections++;
			} else { // ray-triangle intersection behind
				intersections_check++; // cast a second ray to check if starting point is really inside (error correction)
			}
		}
	}
	for(uint i=1u; i<min(intersections, 64u); i++) { // insertion-sort distances
		ushort t = distances[i];
		uint j = i;
		while(j>0u&&distances[j-1u]>t) {
			distances[j] = distances[j-1u];
			j--;
		}
		distances[j] = t;
	}
	bool inside = (intersections%2u)&&(intersections_check%2u);
	const bool set_u = sq(ux)+sq(uy)+sq(uz)+sq(rx)+sq(ry)+sq(rz)>0.0f;
	uint intersection = intersections%2u!=intersections_check%2u; // iterate through column, start with 0 regularly, start with 1 if forward and backward intersection count evenness differs (error correction)
	const uint h0 = direction==0u ? xyz.x : direction==1u ? xyz.y : xyz.z;
	const uint hmesh = h0+(uint)distances[min(intersections-1u, 63u)]; // clamp (intersections-1u) to prevent array out-of-bounds access
	for(uint h=h0; h<=hmax; h++) {
		while(intersection<intersections&&h>h0+(uint)distances[min(intersection, 63u)]) { // clamp intersection to prevent array out-of-bounds access
			inside = !inside; // passed mesh intersection, so switch inside/outside state
			intersection++;
		}
		inside = inside&&(intersection<intersections&&h<hmesh); // point must be outside if there are no more ray-mesh intersections ahead (error correction)
		const uxx n = index((uint3)(direction==0u?h:xyz.x, direction==1u?h:xyz.y, direction==2u?h:xyz.z));
		uchar flagsn = flags[n];
		const float3 p = position(coordinates(n))+offset;
		const float3 u_set = (float3)(ux, uy, uz)+cross((float3)(cx, cy, cz)-p, (float3)(rx, ry, rz));
		if(inside) { // cell is inside of mesh geometry
			flagsn = (flagsn&~TYPE_BO)|flag; // set flag
			if(set_u) store3_u(u, n, u_set); // set solid velocity
		} else { // cell is outside of mesh geometry
			if((flagsn&TYPE_BO)==TYPE_S&&(flagsn&TYPE_XY)==(flag&TYPE_XY)) { // cell was previously marked solid
				// ★ TODO 2 Schritt 4 (12.09.2026) -- LATENT, heute inert, und deshalb hier angesagt statt
				// repariert. Die Zeile darunter vergleicht das ZURUECKGELESENE u EXAKT gegen u_set.
				// Unter U_FP16 ist load_u(store_u(x)) fuer beliebiges x NICHT x (nur das Speicherwort
				// ist ein Fixpunkt, nicht jeder Eingabewert), der Zweig "diese Zelle gehoerte zu dieser
				// Geometrie" feuerte dann NIE und die Solid-nach-Fluid-Rekonstruktion entfiele lautlos.
				// Heute inert, weil alle fuenf Aufrufer set_u == false fahren und u_set damit (0,0,0)
				// ist -- und 0,0f ist unter FP16S ein exakter Fixpunkt (Wort 0x0000). Wer je eine
				// BEWEGTE oder rotierende Geometrie voxelisiert, braucht hier ein Ein-Quant-Epsilon.
				const float3 un = load3_u(u, n); // load previous velocity
				if(un.x==u_set.x&&un.y==u_set.y&&un.z==u_set.z) { // velocity matched: cell belonged to the currently voxelized geometry
)+"#ifndef SPARSE_TILES"+R(
					// FORK: bei SPARSE_TILES ist fi zur Voxelisierungszeit noch der 1-Zell-Platzhalter --
					// finalize_sparse_tiles() legt die echte sparse fi erst DANACH an, weil vorher gar nicht
					// feststeht, welche Tiles tot sind. Ein store_f hierher schriebe ausserhalb des Puffers.
					// Der Verzicht ist folgenlos: initialize() setzt anschliessend ohnehin alle DDFs auf f_eq.
					if(set_u) { // reconstruct DDFs when solid cell is converted to fluid
						uxx j[def_velocity_set]; // neighbor indices
						neighbors(n, j); // calculate neighbor indices
						float feq[def_velocity_set]; // f_equilibrium
						calculate_f_eq(1.0f, un.x, un.y, un.z, feq); // use rhon=1 to prevent mass drift
						store_f(n, feq, fi, j, t TS_A); // write to fi
					}
)+"#endif"+R( // SPARSE_TILES
					flagsn = (flagsn&TYPE_BO)==TYPE_MS ? flagsn&~TYPE_MS : flagsn&~flag; // clear flag
				} // else: don't change cell state
			}
		}
		flags[n] = flagsn;
)+"#ifdef SURFACE"+R(
		mass[n] += massex[n]; // apply distributed excess mass
		massex[n] = 0.0f; // clear excess mass
)+"#endif"+R( // SURFACE
	}
} // voxelize_mesh()

)+R(kernel void unvoxelize_mesh(global uchar* flags, const uchar flag, float x0, float y0, float z0, float x1, float y1, float z1) { // remove voxelized triangle mesh
	const uxx n = get_global_id(0);
	const float3 p = position(coordinates(n))+(float3)(0.5f*(float)((int)def_Nx+2*def_Ox)-0.5f, 0.5f*(float)((int)def_Ny+2*def_Oy)-0.5f, 0.5f*(float)((int)def_Nz+2*def_Oz)-0.5f);
	if(p.x>=x0-1.0f&&p.y>=y0-1.0f&&p.z>=z0-1.0f&&p.x<=x1+1.0f&&p.y<=y1+1.0f&&p.z<=z1+1.0f) flags[n] &= ~flag;
} // unvoxelize_mesh()



// ################################################## graphics code ##################################################

)+"#ifdef GRAPHICS"+R(
)+R(uint3 coordinates_mc(const uxx n) { // disassemble 1D index to 3D coordinates for marching-cubes (n -> x,y,z)
	const uint t = (uint)(n%(uxx)((def_Nx-1u)*(def_Ny-1u)));
	return (uint3)(t%(def_Nx-1u), t/(def_Nx-1u), (uint)(n/(uxx)((def_Nx-1u)*(def_Ny-1u)))); // n = x+(y+z*Ny)*Nx
}
)+R(bool is_halo_mc(const uint3 xyz) {
	return ((def_Dx>1u)&(xyz.x==0u||xyz.x>=def_Nx-2u))||((def_Dy>1u)&(xyz.y==0u||xyz.y>=def_Ny-2u))||((def_Dz>1u)&(xyz.z==0u||xyz.z>=def_Nz-2u)); // halo data is kept up-to-date, so allow using halo data for rendering
}
)+R(void calculate_j8(const uint3 xyz, uxx* j) {
	const uxx x0 = (uxx)  xyz.x; // cube stencil
	const uxx xp = (uxx) (xyz.x+1u);
	const uxx y0 = (uxx)( xyz.y    *def_Nx);
	const uxx yp = (uxx)((xyz.y+1u)*def_Nx);
	const uxx z0 = (uxx)  xyz.z    *(uxx)(def_Ny*def_Nx);
	const uxx zp = (uxx) (xyz.z+1u)*(uxx)(def_Ny*def_Nx);
	j[0] = x0+y0+z0; // 000 // cube stencil
	j[1] = xp+y0+z0; // +00
	j[2] = xp+y0+zp; // +0+
	j[3] = x0+y0+zp; // 00+
	j[4] = x0+yp+z0; // 0+0
	j[5] = xp+yp+z0; // ++0
	j[6] = xp+yp+zp; // +++
	j[7] = x0+yp+zp; // 0++
} // calculate_j8()
)+R(void calculate_j32(const uint3 xyz, uxx* j) {
	const uxx x0 = (uxx)   xyz.x; // cube stencil
	const uxx xp = (uxx)  (xyz.x+1u);
	const uxx y0 = (uxx) ( xyz.y    *def_Nx);
	const uxx yp = (uxx) ((xyz.y+1u)*def_Nx);
	const uxx z0 = (uxx)   xyz.z    *(uxx)(def_Ny*def_Nx);
	const uxx zp = (uxx)  (xyz.z+1u)*(uxx)(def_Ny*def_Nx);
	const uxx xq = (uxx) ((xyz.x       +2u)%def_Nx); // central difference stencil on each cube corner point
	const uxx xm = (uxx) ((xyz.x+def_Nx-1u)%def_Nx);
	const uxx yq = (uxx)(((xyz.y       +2u)%def_Ny)*def_Nx);
	const uxx ym = (uxx)(((xyz.y+def_Ny-1u)%def_Ny)*def_Nx);
	const uxx zq = (uxx) ((xyz.z       +2u)%def_Nz)*(uxx)(def_Ny*def_Nx);
	const uxx zm = (uxx) ((xyz.z+def_Nz-1u)%def_Nz)*(uxx)(def_Ny*def_Nx);
	j[ 0] = x0+y0+z0; // 000 // cube stencil
	j[ 1] = xp+y0+z0; // +00
	j[ 2] = xp+y0+zp; // +0+
	j[ 3] = x0+y0+zp; // 00+
	j[ 4] = x0+yp+z0; // 0+0
	j[ 5] = xp+yp+z0; // ++0
	j[ 6] = xp+yp+zp; // +++
	j[ 7] = x0+yp+zp; // 0++
	j[ 8] = xm+y0+z0; // -00 // central difference stencil on each cube corner point
	j[ 9] = x0+ym+z0; // 0-0
	j[10] = x0+y0+zm; // 00-
	j[11] = xq+y0+z0; // #00
	j[12] = xp+ym+z0; // +-0
	j[13] = xp+y0+zm; // +0-
	j[14] = xq+y0+zp; // #0+
	j[15] = xp+ym+zp; // +-+
	j[16] = xp+y0+zq; // +0#
	j[17] = xm+y0+zp; // -0+
	j[18] = x0+ym+zp; // 0-+
	j[19] = x0+y0+zq; // 00#
	j[20] = xm+yp+z0; // -+0
	j[21] = x0+yq+z0; // 0#0
	j[22] = x0+yp+zm; // 0+-
	j[23] = xq+yp+z0; // #+0
	j[24] = xp+yq+z0; // +#0
	j[25] = xp+yp+zm; // ++-
	j[26] = xq+yp+zp; // #++
	j[27] = xp+yq+zp; // +#+
	j[28] = xp+yp+zq; // ++#
	j[29] = xm+yp+zp; // -++
	j[30] = x0+yq+zp; // 0#+
	j[31] = x0+yp+zq; // 0+#
} // calculate_j32()

)+"#ifndef FORCE_FIELD"+R( // render flags as grid
)+R(kernel void graphics_flags(const global float* camera, global int* bitmap, global int* zbuffer, const global uchar* flags) {
)+"#else"+R( // FORCE_FIELD
)+R(kernel void graphics_flags(const global float* camera, global int* bitmap, global int* zbuffer, const global uchar* flags, const global float* F) {
)+"#endif"+R( // FORCE_FIELD
	const uxx n = get_global_id(0);
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute graphics_flags() on halo
	const uchar flagsn = flags[n]; // cache flags
	const uchar flagsn_bo = flagsn&TYPE_BO; // extract boundary flags
	if(flagsn==0u||flagsn==TYPE_G) return; // don't draw regular fluid cells
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	const uint3 xyz = coordinates(n);
	const float3 p = position(xyz);
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
	uxx x0, xp, xm, y0, yp, ym, z0, zp, zm;
	calculate_indices(n, &x0, &xp, &xm, &y0, &yp, &ym, &z0, &zp, &zm);
	const int c = // coloring scheme
		flagsn_bo==TYPE_S ? COLOR_S : // solid boundary
		((flagsn&TYPE_T)&&flagsn_bo==TYPE_E) ? color_average(COLOR_T, COLOR_E) : // both temperature boundary and equilibrium boundary
		((flagsn&TYPE_T)&&flagsn_bo==TYPE_MS) ? color_average(COLOR_T, COLOR_M) : // both temperature boundary and moving boundary
		flagsn&TYPE_T ? COLOR_T : // temperature boundary
		flagsn_bo==TYPE_E ? COLOR_E : // equilibrium boundary
		flagsn_bo==TYPE_MS ? COLOR_M : // moving boundary
		flagsn&TYPE_F ? COLOR_F : // fluid
		flagsn&TYPE_I ? COLOR_I : // interface
		flagsn&TYPE_X ? COLOR_X : // reserved type X
		flagsn&TYPE_Y ? COLOR_Y : // reserved type Y
		COLOR_0; // regular or gas cell
	//draw_point(p, c, camera_cache, bitmap, zbuffer); // draw one pixel for every boundary cell
	uxx t;
	t = xp+y0+z0; const bool not_xp = xyz.x<def_Nx-1u && flagsn==flags[t] && !is_halo(t); // +00
	t = xm+y0+z0; const bool not_xm = xyz.x>       0u && flagsn==flags[t] && !is_halo(t); // -00
	t = x0+yp+z0; const bool not_yp = xyz.y<def_Ny-1u && flagsn==flags[t] && !is_halo(t); // 0+0
	t = x0+ym+z0; const bool not_ym = xyz.y>       0u && flagsn==flags[t] && !is_halo(t); // 0-0
	t = x0+y0+zp; const bool not_zp = xyz.z<def_Nz-1u && flagsn==flags[t] && !is_halo(t); // 00+
	t = x0+y0+zm; const bool not_zm = xyz.z>       0u && flagsn==flags[t] && !is_halo(t); // 00-
	const float3 p0 = (float3)(p.x-0.5f, p.y-0.5f, p.z-0.5f); // ---
	const float3 p1 = (float3)(p.x+0.5f, p.y+0.5f, p.z+0.5f); // +++
	const float3 p2 = (float3)(p.x-0.5f, p.y-0.5f, p.z+0.5f); // --+
	const float3 p3 = (float3)(p.x+0.5f, p.y+0.5f, p.z-0.5f); // ++-
	const float3 p4 = (float3)(p.x-0.5f, p.y+0.5f, p.z-0.5f); // -+-
	const float3 p5 = (float3)(p.x+0.5f, p.y-0.5f, p.z+0.5f); // +-+
	const float3 p6 = (float3)(p.x+0.5f, p.y-0.5f, p.z-0.5f); // +--
	const float3 p7 = (float3)(p.x-0.5f, p.y+0.5f, p.z+0.5f); // -++
	if(!(not_xm||not_ym)) draw_line(p0, p2, c, camera_cache, bitmap, zbuffer); // to draw the entire surface, replace || by &&
	if(!(not_xm||not_zm)) draw_line(p0, p4, c, camera_cache, bitmap, zbuffer);
	if(!(not_ym||not_zm)) draw_line(p0, p6, c, camera_cache, bitmap, zbuffer);
	if(!(not_xp||not_yp)) draw_line(p1, p3, c, camera_cache, bitmap, zbuffer);
	if(!(not_xp||not_zp)) draw_line(p1, p5, c, camera_cache, bitmap, zbuffer);
	if(!(not_yp||not_zp)) draw_line(p1, p7, c, camera_cache, bitmap, zbuffer);
	if(!(not_ym||not_zp)) draw_line(p2, p5, c, camera_cache, bitmap, zbuffer);
	if(!(not_xm||not_zp)) draw_line(p2, p7, c, camera_cache, bitmap, zbuffer);
	if(!(not_yp||not_zm)) draw_line(p3, p4, c, camera_cache, bitmap, zbuffer);
	if(!(not_xp||not_zm)) draw_line(p3, p6, c, camera_cache, bitmap, zbuffer);
	if(!(not_xm||not_yp)) draw_line(p4, p7, c, camera_cache, bitmap, zbuffer);
	if(!(not_xp||not_ym)) draw_line(p5, p6, c, camera_cache, bitmap, zbuffer);
)+"#ifdef FORCE_FIELD"+R(
	if(flagsn_bo==TYPE_S) {
		const float3 Fn = def_scale_F*load3(F, n); // LATENT (Gross-Audit): voll-domaeniger Index -- unter F-BBox OOB; GRAPHICS ist wegkompiliert, bei Wiederbelebung load3_F verwenden
		const float Fnl = length(Fn);
		if(Fnl>0.0f) {
			const int c = colorscale_iron(Fnl); // color boundaries depending on the force on them
			draw_line(p, p+Fn, c, camera_cache, bitmap, zbuffer); // draw colored force vectors
		}
	}
)+"#endif"+R( // FORCE_FIELD
}

)+"#ifndef FORCE_FIELD"+R( // render solid boundaries with marching-cubes
)+R(kernel void graphics_flags_mc(const global float* camera, global int* bitmap, global int* zbuffer, const global uchar* flags) {
)+"#else"+R( // FORCE_FIELD
)+R(kernel void graphics_flags_mc(const global float* camera, global int* bitmap, global int* zbuffer, const global uchar* flags, const global float* F) {
)+"#endif"+R( // FORCE_FIELD
	const uxx n = get_global_id(0);
)+"#if LSF>0u"+R( // use local memory
	const uxx n_global = n/(LSF*LSF*LSF);
	const uint n_local = (uint)(n%(LSF*LSF*LSF));
	const uint t_global = (uint)(n_global%(uxx)(((def_Nx+LSF-2u)/LSF)*((def_Ny+LSF-2u)/LSF))); // -1u for coordinates_mc, +LSF-1u for always rounding up
	const uint3 xyz_global = (uint3)(t_global%((def_Nx+LSF-2u)/LSF), t_global/((def_Nx+LSF-2u)/LSF), (uint)(n_global/(uxx)(((def_Nx+LSF-2u)/LSF)*((def_Ny+LSF-2u)/LSF)))); // n = x+(y+z*Ny)*Nx
	const uint t_local = n_local%(LSF*LSF);
	const uint3 xyz_local = (uint3)(t_local%LSF, t_local/LSF, n_local/(LSF*LSF)); // n = x+(y+z*Ny)*Nx
	const uint3 xyz = LSF*xyz_global+xyz_local;
	local uchar flags_cache[(LSF+1u)*(LSF+1u)*(LSF+1u)]; // for LSF==4: 4x4x4 cells with [0,+1] halo = 5x5x5 cells (125 Byte) to load in cache
)+"#ifdef FORCE_FIELD"+R(
	local float3 F_cache[(LSF+1u)*(LSF+1u)*(LSF+1u)]; // for LSF==4: 4x4x4 cells with [0,+1] halo = 5x5x5 cells (1500 Byte) to load in cache
)+"#endif"+R( // FORCE_FIELD
	const uint loads_per_thread = ((LSF+1u)*(LSF+1u)*(LSF+1u)+LSF*LSF*LSF-1u)/(LSF*LSF*LSF); // number of grid cells each thread has to load (for LSF==4: 2, for LSF==8: 2)
	for(uint c=0u; c<loads_per_thread; c++) {
		const uint n_local_c = c*(LSF*LSF*LSF)+n_local; // SoA (>2x faster on GPUs)
		if(n_local_c<(LSF+1u)*(LSF+1u)*(LSF+1u)) {
			const uint t_local_c = n_local_c%((LSF+1u)*(LSF+1u));
			const uint3 xyz_local_c = (uint3)(t_local_c%(LSF+1u), t_local_c/(LSF+1u), n_local_c/((LSF+1u)*(LSF+1u))); // n = x+(y+z*Ny)*Nx
			const uint3 xyz_global_c = (LSF*xyz_global+xyz_local_c)%(uint3)(def_Nx, def_Ny, def_Nz); // apply periodic boundaries
			const uxx n_global_c = index(xyz_global_c);
			const uchar flags_nc = (flags[n_global_c]&TYPE_BO)==TYPE_S; // load flags from global memory into local memory
			flags_cache[n_local_c] = flags_nc;
)+"#ifdef FORCE_FIELD"+R(
			F_cache[n_local_c] = (flags_nc ? load3(F, n_global_c) : (float3)(0.0f, 0.0f, 0.0f)); // load F from global memory into local memory
)+"#endif"+R( // FORCE_FIELD
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(xyz.x>=def_Nx-1u||xyz.y>=def_Ny-1u||xyz.z>=def_Nz-1u||is_halo_mc(xyz)) return; // don't execute graphics_flags_mc() on marching-cubes halo
)+"#else"+R( // do not use local memory
	if(n>=(uxx)(def_Nx-1u)*(uxx)(def_Ny-1u)*(uxx)(def_Nz-1u)) return;
	const uint3 xyz = coordinates_mc(n);
	if(is_halo_mc(xyz)) return; // don't execute graphics_flags_mc() on marching-cubes halo
)+"#endif"+R( // do not use local memory
	const float3 p = position(xyz);
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
	bool v[8];
)+"#ifdef FORCE_FIELD"+R(
	float3 Fj[8];
)+"#endif"+R( // FORCE_FIELD
)+"#if LSF>0u"+R( // use local memory
	{ // load 8-cell stencil from cache
		const uint x0=xyz_local.x                    , xp=x0+               1u  ;
		const uint y0=xyz_local.y*          (LSF+1u) , yp=y0+          (LSF+1u) ;
		const uint z0=xyz_local.z*((LSF+1u)*(LSF+1u)), zp=z0+((LSF+1u)*(LSF+1u));
		v[0] = flags_cache[x0+y0+z0]; // 000 // cube stencil
		v[1] = flags_cache[xp+y0+z0]; // +00
		v[2] = flags_cache[xp+y0+zp]; // +0+
		v[3] = flags_cache[x0+y0+zp]; // 00+
		v[4] = flags_cache[x0+yp+z0]; // 0+0
		v[5] = flags_cache[xp+yp+z0]; // ++0
		v[6] = flags_cache[xp+yp+zp]; // +++
		v[7] = flags_cache[x0+yp+zp]; // 0++
)+"#ifdef FORCE_FIELD"+R(
		Fj[0] = F_cache[x0+y0+z0]; // 000 // cube stencil
		Fj[1] = F_cache[xp+y0+z0]; // +00
		Fj[2] = F_cache[xp+y0+zp]; // +0+
		Fj[3] = F_cache[x0+y0+zp]; // 00+
		Fj[4] = F_cache[x0+yp+z0]; // 0+0
		Fj[5] = F_cache[xp+yp+z0]; // ++0
		Fj[6] = F_cache[xp+yp+zp]; // +++
		Fj[7] = F_cache[x0+yp+zp]; // 0++
)+"#endif"+R( // FORCE_FIELD
	}
)+"#else"+R( // do not use local memory
	uxx j[8];
	calculate_j8(xyz, j);
	for(uint i=0u; i<8u; i++) v[i] = (flags[j[i]]&TYPE_BO)==TYPE_S;
)+"#ifdef FORCE_FIELD"+R(
	for(uint i=0u; i<8u; i++) Fj[i] = (v[i] ? load3(F, j[i]) : (float3)(0.0f, 0.0f, 0.0f));
)+"#endif"+R( // FORCE_FIELD
)+"#endif"+R( // do not use local memory
	float3 triangles[15]; // maximum of 5 triangles with 3 vertices each
	const uint tn = marching_cubes_halfway(v, triangles); // run marching cubes algorithm
	if(tn==0u) return;
	for(uint i=0u; i<tn; i++) {
		const float3 p0 = triangles[3u*i   ];
		const float3 p1 = triangles[3u*i+1u];
		const float3 p2 = triangles[3u*i+2u];
		int c0=0xDFDFDF, c1=0xDFDFDF, c2=0xDFDFDF;
)+"#ifdef FORCE_FIELD"+R(
		const float3 normal = normalize(cross(p1-p0, p2-p0));
		c0 = colorscale_twocolor(0.5f+def_scale_F*dot(trilinear3(p0, Fj), normal));
		c1 = colorscale_twocolor(0.5f+def_scale_F*dot(trilinear3(p1, Fj), normal));
		c2 = colorscale_twocolor(0.5f+def_scale_F*dot(trilinear3(p2, Fj), normal));
)+"#else"+R( // FORCE_FIELD
		const float3 normal = cross(p1-p0, p2-p0); // no normalize needed for shading()
)+"#endif"+R( // FORCE_FIELD
		c0 = shading(c0, p+p0, normal, camera_cache);
		c1 = shading(c1, p+p1, normal, camera_cache);
		c2 = shading(c2, p+p2, normal, camera_cache);
		draw_triangle_interpolated(p+p0, p+p1, p+p2, c0, c1, c2, camera_cache, bitmap, zbuffer); // draw triangle with interpolated colors
	}
}

)+"#ifndef TEMPERATURE"+R(
)+R(kernel void graphics_field(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u, const global uchar* flags) {
)+"#else"+R( // TEMPERATURE
)+R(kernel void graphics_field(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u, const global uchar* flags, const global float* T) {
)+"#endif"+R( // TEMPERATURE
	const uxx n = get_global_id(0);
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute graphics_field() on halo
	const uint3 xyz = coordinates(n);
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	const float3 p = position(xyz);
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
)+"#ifndef MOVING_BOUNDARIES"+R(
	if(flags[n]&(TYPE_S|TYPE_E|TYPE_I|TYPE_G)) return;
)+"#else"+R( // MOVING_BOUNDARIES
	if(flags[n]&(TYPE_I|TYPE_G)) return;
)+"#endif"+R( // MOVING_BOUNDARIES
	const float3 un = load3(u, n); // cache velocity
	const float ul = length(un);
	if(def_scale_u*ul<0.1f) return; // don't draw lattice points where the velocity is lower than this threshold
	int c = 0; // coloring
	switch(field_mode) {
		case 0: c = colorscale_rainbow(def_scale_u*ul); break; // coloring by velocity
		case 1: c = colorscale_twocolor(0.5f+def_scale_rho*(rho[n]-1.0f)); break; // coloring by density
)+"#ifdef TEMPERATURE"+R(
		case 2: c = colorscale_iron(0.5f+def_scale_T*(T[n]-def_T_avg)); break; // coloring by temperature
)+"#endif"+R( // TEMPERATURE
	}
	draw_line(p-(0.5f/ul)*un, p+(0.5f/ul)*un, c, camera_cache, bitmap, zbuffer);
}

)+"#ifndef TEMPERATURE"+R(
)+R(int ray_grid_traverse_sum(const int background_color, const ray r, const uint Nx, const uint Ny, const uint Nz, const int field_mode, const global float* rho, const global float* u, const global uchar* flags) {
)+"#else"+R( // TEMPERATURE
)+R(int ray_grid_traverse_sum(const int background_color, const ray r, const uint Nx, const uint Ny, const uint Nz, const int field_mode, const global float* rho, const global float* u, const global uchar* flags, const global float* T) {
)+"#endif"+R( // TEMPERATURE
	float sum = 0.0f;
	float traversed_cells_weighted = 0.0f;
	uint traversed_cells = 0u;
	const float3 p = (float3)(r.origin.x+0.5f*(float)Nx, r.origin.y+0.5f*(float)Ny, r.origin.z+0.5f*(float)Nz); // start point
	const int dx=(int)sign(r.direction.x), dy=(int)sign(r.direction.y), dz=(int)sign(r.direction.z); // fast ray-grid-traversal
	int3 xyz = (int3)((int)floor(p.x), (int)floor(p.y), (int)floor(p.z));
	const float fxa=p.x-floor(p.x), fya=p.y-floor(p.y), fza=p.z-floor(p.z);
	const float tdx = 1.0f/fmax(fabs(r.direction.x), 1E-6f);
	const float tdy = 1.0f/fmax(fabs(r.direction.y), 1E-6f);
	const float tdz = 1.0f/fmax(fabs(r.direction.z), 1E-6f);
	float tmx = tdx*(dx>0 ? 1.0f-fxa : dx<0 ? fxa : 0.0f);
	float tmy = tdy*(dy>0 ? 1.0f-fya : dy<0 ? fya : 0.0f);
	float tmz = tdz*(dz>0 ? 1.0f-fza : dz<0 ? fza : 0.0f);
	int color = 0;
	switch(field_mode) {
		case 0: // coloring by velocity
			while(traversed_cells<Nx+Ny+Nz) { // limit number of traversed cells to space diagonal
				if(tmx<tmy) { if(tmx<tmz) { xyz.x += dx; tmx += tdx; } else { xyz.z += dz; tmz += tdz; } }
				else /****/ { if(tmy<tmz) { xyz.y += dy; tmy += tdy; } else { xyz.z += dz; tmz += tdz; } }
				if(xyz.x<0 || xyz.y<0 || xyz.z<0 || xyz.x>=(int)Nx || xyz.y>=(int)Ny || xyz.z>=(int)Nz) break; // out of simulation box
				const uxx n = index((uint3)((uint)clamp(xyz.x, 0, (int)Nx-1), (uint)clamp(xyz.y, 0, (int)Ny-1), (uint)clamp(xyz.z, 0, (int)Nz-1)));
				if(!(flags[n]&(TYPE_S|TYPE_E|TYPE_G))) {
					const float un = length(load3(u, n));
					const float weight = fmin(un, fabs(un-0.5f/def_scale_u));
					sum = fma(weight, un, sum);
					traversed_cells_weighted += weight;
				}
				traversed_cells++;
			}
			color = colorscale_rainbow(def_scale_u*sum/traversed_cells_weighted);
			traversed_cells_weighted *= 2.0f*def_scale_u;
			break;
		case 1: // coloring by density
			while(traversed_cells<Nx+Ny+Nz) { // limit number of traversed cells to space diagonal
				if(tmx<tmy) { if(tmx<tmz) { xyz.x += dx; tmx += tdx; } else { xyz.z += dz; tmz += tdz; } }
				else /****/ { if(tmy<tmz) { xyz.y += dy; tmy += tdy; } else { xyz.z += dz; tmz += tdz; } }
				if(xyz.x<0 || xyz.y<0 || xyz.z<0 || xyz.x>=(int)Nx || xyz.y>=(int)Ny || xyz.z>=(int)Nz) break; // out of simulation box
				const uxx n = index((uint3)((uint)clamp(xyz.x, 0, (int)Nx-1), (uint)clamp(xyz.y, 0, (int)Ny-1), (uint)clamp(xyz.z, 0, (int)Nz-1)));
				if(!(flags[n]&(TYPE_S|TYPE_E|TYPE_G))) {
					const float rhon = rho[n];
					const float weight = fabs(rhon-1.0f);
					sum = fma(weight, rhon, sum);
					traversed_cells_weighted += weight;
				}
				traversed_cells++;
			}
			color = colorscale_twocolor(0.5f+def_scale_rho*(sum/traversed_cells_weighted-1.0f));
			traversed_cells_weighted *= def_scale_rho;
			break;
)+"#ifdef TEMPERATURE"+R(
		case 2: // coloring by temperature
			while(traversed_cells<Nx+Ny+Nz) { // limit number of traversed cells to space diagonal
				if(tmx<tmy) { if(tmx<tmz) { xyz.x += dx; tmx += tdx; } else { xyz.z += dz; tmz += tdz; } }
				else /****/ { if(tmy<tmz) { xyz.y += dy; tmy += tdy; } else { xyz.z += dz; tmz += tdz; } }
				if(xyz.x<0 || xyz.y<0 || xyz.z<0 || xyz.x>=(int)Nx || xyz.y>=(int)Ny || xyz.z>=(int)Nz) break; // out of simulation box
				const uxx n = index((uint3)((uint)clamp(xyz.x, 0, (int)Nx-1), (uint)clamp(xyz.y, 0, (int)Ny-1), (uint)clamp(xyz.z, 0, (int)Nz-1)));
				if(!(flags[n]&(TYPE_S|TYPE_E|TYPE_G))) {
					const float Tn = T[n];
					const float weight = sq(Tn-def_T_avg);
					sum = fma(weight, Tn, sum);
					traversed_cells_weighted += weight;
				}
				traversed_cells++;
			}
			color = colorscale_iron(0.5f+def_scale_T*(sum/traversed_cells_weighted-def_T_avg));
			traversed_cells_weighted *= sq(4.0f*def_scale_T);
			break;
)+"#endif"+R( // TEMPERATURE
	}
	const float opacity = clamp((traversed_cells_weighted-1.0f)/(float)traversed_cells, 0.0f, 1.0f);
	return color_mix(color, background_color, opacity);
}

)+"#ifndef TEMPERATURE"+R(
)+R(kernel void graphics_field_rt(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u, const global uchar* flags) {
)+"#else"+R( // TEMPERATURE
)+R(kernel void graphics_field_rt(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u, const global uchar* flags, const global float* T) {
)+"#endif"+R( // TEMPERATURE
	const uint gid = get_global_id(0); // workgroup size alignment is critical
	const uint lid = get_local_id(0); // make workgropus not horizontal stripes of pixels, but 8x8 rectangular (close to square) tiles
	const uint lsi = get_local_size(0); // (50% performance boost due to more coalesced memory access)
	const uint tile_width=8u, tile_height=lsi/tile_width, tiles_x=def_screen_width/tile_width;
	const int lx=lid%tile_width, ly=lid/tile_width;
	const int tx=(gid/lsi)%tiles_x, ty=(gid/lsi)/tiles_x;
	const int x=tx*tile_width+lx, y=ty*tile_height+ly;
	const uint n = x+y*def_screen_width;
	float camera_cache[15]; // cache parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	ray camray = get_camray(x, y, camera_cache);
	const float distance = intersect_cuboid(camray, (float3)(0.0f, 0.0f, 0.0f), (float)def_Nx, (float)def_Ny, (float)def_Nz);
	if(distance==-1.0f) return;
	camray.origin = camray.origin+fmax(distance+0.005f, 0.005f)*camray.direction;
)+"#ifndef TEMPERATURE"+R(
	bitmap[n] = ray_grid_traverse_sum(bitmap[n], camray, def_Nx, def_Ny, def_Nz, field_mode, rho, u, flags);
)+"#else"+R( // TEMPERATURE
	bitmap[n] = ray_grid_traverse_sum(bitmap[n], camray, def_Nx, def_Ny, def_Nz, field_mode, rho, u, flags, T);
)+"#endif"+R( // TEMPERATURE
}

)+"#ifndef TEMPERATURE"+R(
)+R(kernel void graphics_field_slice(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const int slice_mode, const int slice_x, const int slice_y, const int slice_z, const global float* rho, const global float* u, const global uchar* flags) {
)+"#else"+R( // TEMPERATURE
)+R(kernel void graphics_field_slice(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const int slice_mode, const int slice_x, const int slice_y, const int slice_z, const global float* rho, const global float* u, const global uchar* flags, const global float* T) {
)+"#endif"+R( // TEMPERATURE
	const uint a = get_global_id(0);
	const uint direction = (uint)clamp(slice_mode-1, 0, 2);
	if(a>=get_area(direction)||slice_mode<1||slice_mode>3||(slice_mode==1&&(slice_x<0||slice_x>=(int)def_Nx))||(slice_mode==2&&(slice_y<0||slice_y>=(int)def_Ny))||(slice_mode==3&&(slice_z<0||slice_z>=(int)def_Nz))) return;
	uint3 xyz00, xyz01, xyz10, xyz11;
	float3 normal;
	switch(direction) {
		case 0u: xyz00 = (uint3)((uint)slice_x, a%def_Ny, a/def_Ny); if(xyz00.y>=def_Ny-1u||xyz00.z>=def_Nz-1u) return; xyz01 = xyz00+(uint3)(0u, 0u, 1u); xyz10 = xyz00+(uint3)(0u, 1u, 0u); xyz11 = xyz00+(uint3)(0u, 1u, 1u); normal = (float3)(1.0f, 0.0f, 0.0f); break;
		case 1u: xyz00 = (uint3)(a/def_Nz, (uint)slice_y, a%def_Nz); if(xyz00.x>=def_Nx-1u||xyz00.z>=def_Nz-1u) return; xyz01 = xyz00+(uint3)(0u, 0u, 1u); xyz10 = xyz00+(uint3)(1u, 0u, 0u); xyz11 = xyz00+(uint3)(1u, 0u, 1u); normal = (float3)(0.0f, 1.0f, 0.0f); break;
		case 2u: xyz00 = (uint3)(a%def_Nx, a/def_Nx, (uint)slice_z); if(xyz00.x>=def_Nx-1u||xyz00.y>=def_Ny-1u) return; xyz01 = xyz00+(uint3)(0u, 1u, 0u); xyz10 = xyz00+(uint3)(1u, 0u, 0u); xyz11 = xyz00+(uint3)(1u, 1u, 0u); normal = (float3)(0.0f, 0.0f, 1.0f); break;
	}
	const float3 p00=position(xyz00), p01=position(xyz01), p10=position(xyz10), p11=position(xyz11), p=0.25f*(p00+p01+p10+p11);
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
	const uxx n00=index(xyz00), n01=index(xyz01), n10=index(xyz10), n11=index(xyz11);
	bool d00=true, d01=true, d10=true, d11=true;
)+"#ifdef SURFACE"+R(
	d00 = flags[n00]&(TYPE_F|TYPE_I); // only draw fluid or interface cells
	d01 = flags[n01]&(TYPE_F|TYPE_I);
	d10 = flags[n10]&(TYPE_F|TYPE_I);
	d11 = flags[n11]&(TYPE_F|TYPE_I);
	if((int)d00+(int)d01+(int)d10+(int)d11<3) return;
)+"#endif"+R( // SURFACE
	int c00=0, c01=0, c10=0, c11=0;
	switch(field_mode) {
		case 0: // coloring by velocity
			c00 = colorscale_rainbow(def_scale_u*length(load3(u, n00)));
			c01 = colorscale_rainbow(def_scale_u*length(load3(u, n01)));
			c10 = colorscale_rainbow(def_scale_u*length(load3(u, n10)));
			c11 = colorscale_rainbow(def_scale_u*length(load3(u, n11)));
			break;
		case 1: // coloring by density
			c00 = colorscale_twocolor(0.5f+def_scale_rho*(rho[n00]-1.0f));
			c01 = colorscale_twocolor(0.5f+def_scale_rho*(rho[n01]-1.0f));
			c10 = colorscale_twocolor(0.5f+def_scale_rho*(rho[n10]-1.0f));
			c11 = colorscale_twocolor(0.5f+def_scale_rho*(rho[n11]-1.0f));
			break;
)+"#ifdef TEMPERATURE"+R(
		case 2: // coloring by temperature
			c00 = colorscale_iron(0.5f+def_scale_T*(T[n00]-def_T_avg));
			c01 = colorscale_iron(0.5f+def_scale_T*(T[n01]-def_T_avg));
			c10 = colorscale_iron(0.5f+def_scale_T*(T[n10]-def_T_avg));
			c11 = colorscale_iron(0.5f+def_scale_T*(T[n11]-def_T_avg));
			break;
)+"#endif"+R( // TEMPERATURE
	}
	c00 = shading(c00, p00, normal, camera_cache);
	c01 = shading(c01, p01, normal, camera_cache);
	c10 = shading(c10, p10, normal, camera_cache);
	c11 = shading(c11, p11, normal, camera_cache);
	const int c = color_average(color_average(c00, c11), color_average(c01, c10));
	if(d00&&d01) draw_triangle_interpolated(p00, p01, p, c00, c01, c, camera_cache, bitmap, zbuffer);
	if(d01&&d11) draw_triangle_interpolated(p01, p11, p, c01, c11, c, camera_cache, bitmap, zbuffer);
	if(d11&&d10) draw_triangle_interpolated(p11, p10, p, c11, c10, c, camera_cache, bitmap, zbuffer);
	if(d10&&d00) draw_triangle_interpolated(p10, p00, p, c10, c00, c, camera_cache, bitmap, zbuffer);
}

)+"#ifndef TEMPERATURE"+R(
)+R(kernel void graphics_streamline(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const int slice_mode, const int slice_x, const int slice_y, const int slice_z, const global float* rho, const global float* u, const global uchar* flags) {
)+"#else"+R( // TEMPERATURE
)+R(kernel void graphics_streamline(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const int slice_mode, const int slice_x, const int slice_y, const int slice_z, const global float* rho, const global float* u, const global uchar* flags, const global float* T) {
)+"#endif"+R( // TEMPERATURE
	const uxx n = get_global_id(0);
	const float3 ps = (float3)((float)slice_x+0.5f-0.5f*(float)def_Nx, (float)slice_y+0.5f-0.5f*(float)def_Ny, (float)slice_z+0.5f-0.5f*(float)def_Nz);
)+"#ifndef D2Q9"+R(
	if(n>=(uxx)(def_Nx/def_streamline_sparse)*(uxx)(def_Ny/def_streamline_sparse)*(uxx)(def_Nz/def_streamline_sparse)) return;
	const uint z = (uint)(n/(uxx)((def_Nx/def_streamline_sparse)*(def_Ny/def_streamline_sparse))); // disassemble 1D index to 3D coordinates
	const uint t = (uint)(n%(uxx)((def_Nx/def_streamline_sparse)*(def_Ny/def_streamline_sparse)));
	const uint y = (uint)(t/(def_Nx/def_streamline_sparse));
	const uint x = (uint)(t%(def_Nx/def_streamline_sparse));
	float3 p = (float)def_streamline_sparse*((float3)((float)x+0.5f, (float)y+0.5f, (float)z+0.5f))-0.5f*((float3)((float)def_Nx, (float)def_Ny, (float)def_Nz));
	const bool rx=fabs(p.x-ps.x)>0.5f*(float)def_streamline_sparse, ry=fabs(p.y-ps.y)>0.5f*(float)def_streamline_sparse, rz=fabs(p.z-ps.z)>0.5f*(float)def_streamline_sparse;
)+"#else"+R( // D2Q9
	if(n>=(def_Nx/def_streamline_sparse)*(def_Ny/def_streamline_sparse)) return;
	const uint y = (uint)(n/(uxx)(def_Nx/def_streamline_sparse)); // disassemble 1D index to 3D coordinates
	const uint x = (uint)(n%(uxx)(def_Nx/def_streamline_sparse));
	float3 p = ((float3)((float)def_streamline_sparse*((float)x+0.5f), (float)def_streamline_sparse*((float)y+0.5f), 0.5f))-0.5f*((float3)((float)def_Nx, (float)def_Ny, (float)def_Nz));
	const bool rx=fabs(p.x-ps.x)>0.5f*(float)def_streamline_sparse, ry=fabs(p.y-ps.y)>0.5f*(float)def_streamline_sparse, rz=true;
)+"#endif"+R( // D2Q9
	if((slice_mode==1&&rx)||(slice_mode==2&&ry)||(slice_mode==3&&rz)||(slice_mode==4&&rx&&rz)||(slice_mode==5&&rx&&ry&&rz)||(slice_mode==6&&ry&&rz)||(slice_mode==7&&rx&&ry)) return;
	if((slice_mode==1||slice_mode==5||slice_mode==4||slice_mode==7)&!rx) p.x = ps.x; // snap streamline position to slice position
	if((slice_mode==2||slice_mode==5||slice_mode==6||slice_mode==7)&!ry) p.y = ps.y;
	if((slice_mode==3||slice_mode==5||slice_mode==4||slice_mode==6)&!rz) p.z = ps.z;
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	const float hLx=0.5f*(float)(def_Nx-2u*(def_Dx>1u)), hLy=0.5f*(float)(def_Ny-2u*(def_Dy>1u)), hLz=0.5f*(float)(def_Nz-2u*(def_Dz>1u));
	//draw_circle(p, 0.5f*def_streamline_sparse, 0xFFFFFF, camera_cache, bitmap, zbuffer);
	for(float dt=-1.0f; dt<=1.0f; dt+=2.0f) { // integrate forward and backward in time
		float3 p0, p1=p;
		for(uint l=0u; l<def_streamline_length/2u; l++) {
			const uint x = (uint)(p1.x+1.5f*(float)def_Nx)%def_Nx;
			const uint y = (uint)(p1.y+1.5f*(float)def_Ny)%def_Ny;
			const uint z = (uint)(p1.z+1.5f*(float)def_Nz)%def_Nz;
			const uxx n = (uxx)x+(uxx)(y+z*def_Ny)*(uxx)def_Nx;
			if(flags[n]&(TYPE_S|TYPE_E|TYPE_I|TYPE_G)) return;
			const float3 un = load3(u, n); // interpolate_u(u, p1)
			const float ul = length(un);
			p0 = p1;
			p1 += (dt/ul)*un; // integrate forward in time
			if(def_scale_u*ul<0.1f||p1.x<-hLx||p1.x>hLx||p1.y<-hLy||p1.y>hLy||p1.z<-hLz||p1.z>hLz) break;
			int c = 0; // coloring
			switch(field_mode) {
				case 0: c = colorscale_rainbow(def_scale_u*ul); break; // coloring by velocity
				case 1: c = colorscale_twocolor(0.5f+def_scale_rho*(rho[n]-1.0f)); break; // coloring by density
)+"#ifdef TEMPERATURE"+R(
				case 2: c = colorscale_iron(0.5f+def_scale_T*(T[n]-def_T_avg)); break; // coloring by temperature
)+"#endif"+R( // TEMPERATURE
			}
			draw_line(p0, p1, c, camera_cache, bitmap, zbuffer);
		}
	}
}

)+"#ifndef TEMPERATURE"+R(
)+R(kernel void graphics_q_field(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u, const global uchar* flags) {
)+"#else"+R( // TEMPERATURE
)+R(kernel void graphics_q_field(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u, const global uchar* flags, const global float* T) {
)+"#endif"+R( // TEMPERATURE
	const uxx n = get_global_id(0);
	if(n>=(uxx)def_N||is_halo(n)) return; // don't execute graphics_q_field() on halo
	if(flags[n]&(TYPE_S|TYPE_E|TYPE_I|TYPE_G)) return;
	const float3 p = position(coordinates(n));
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
	float3 un = load3(u, n); // cache velocity
	const float ul = length(un);
	const float Q = calculate_Q(n, u);
	if(Q<def_scale_Q_min||ul==0.0f) return; // don't draw lattice points where the velocity is very low
	int c = 0; // coloring
	switch(field_mode) {
		case 0: c = colorscale_rainbow(def_scale_u*ul); break; // coloring by velocity
		case 1: c = colorscale_twocolor(0.5f+def_scale_rho*(rho[n]-1.0f)); break; // coloring by density
)+"#ifdef TEMPERATURE"+R(
		case 2: c = colorscale_iron(0.5f+def_scale_T*(T[n]-def_T_avg)); break; // coloring by temperature
)+"#endif"+R( // TEMPERATURE
	}
	draw_line(p-(0.5f/ul)*un, p+(0.5f/ul)*un, c, camera_cache, bitmap, zbuffer);
}

)+R(kernel void graphics_q)+"("+R(const global float* camera, global int* bitmap, global int* zbuffer, const int field_mode, const global float* rho, const global float* u // ) {
)+"#ifdef SURFACE"+R(
	, const global uchar* flags // argument order is important
)+"#endif"+R( // SURFACE
)+"#ifdef TEMPERATURE"+R(
	, const global float* T // argument order is important
)+"#endif"+R( // TEMPERATURE
)+") {"+R( // graphics_q()
	const uxx n = get_global_id(0);
)+"#if LSQ>0u"+R( // use local memory
	const uxx n_global = n/(LSQ*LSQ*LSQ);
	const uint n_local = (uint)(n%(LSQ*LSQ*LSQ));
	const uint t_global = (uint)(n_global%(uxx)(((def_Nx+LSQ-2u)/LSQ)*((def_Ny+LSQ-2u)/LSQ))); // -1u for coordinates_mc, +LSQ-1u for always rounding up
	const uint3 xyz_global = (uint3)(t_global%((def_Nx+LSQ-2u)/LSQ), t_global/((def_Nx+LSQ-2u)/LSQ), (uint)(n_global/(uxx)(((def_Nx+LSQ-2u)/LSQ)*((def_Ny+LSQ-2u)/LSQ)))); // n = x+(y+z*Ny)*Nx
	const uint t_local = n_local%(LSQ*LSQ);
	const uint3 xyz_local = (uint3)(t_local%LSQ, t_local/LSQ, n_local/(LSQ*LSQ)); // n = x+(y+z*Ny)*Nx
	const uint3 xyz = LSQ*xyz_global+xyz_local;
	local float3 u_cache[(LSQ+3u)*(LSQ+3u)*(LSQ+3u)]; // for LSQ==8: 8x8x8 cells with [-1,+2] halo = 11x11x11 cells (15972 Byte) to load in cache
	const uint loads_per_thread = ((LSQ+3u)*(LSQ+3u)*(LSQ+3u)+LSQ*LSQ*LSQ-1u)/(LSQ*LSQ*LSQ); // number of grid cells each thread has to load (for LSQ==4: 6, for LSQ==8: 3)
	for(uint c=0u; c<loads_per_thread; c++) {
		const uint n_local_c = c*(LSQ*LSQ*LSQ)+n_local; // SoA (>2x faster on GPUs)
		if(n_local_c<(LSQ+3u)*(LSQ+3u)*(LSQ+3u)) {
			const uint t_local_c = n_local_c%((LSQ+3u)*(LSQ+3u));
			const uint3 xyz_local_c = (uint3)(t_local_c%(LSQ+3u), t_local_c/(LSQ+3u), n_local_c/((LSQ+3u)*(LSQ+3u))); // n = x+(y+z*Ny)*Nx
			const uint3 xyz_global_c = (LSQ*xyz_global+xyz_local_c+(uint3)(def_Nx-1u, def_Ny-1u, def_Nz-1u))%(uint3)(def_Nx, def_Ny, def_Nz); // shift by -1 cell and apply periodic boundaries
			u_cache[n_local_c] = load3(u, index(xyz_global_c)); // load u from global memory into local memory
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(xyz.x>=def_Nx-1u||xyz.y>=def_Ny-1u||xyz.z>=def_Nz-1u||is_halo_mc(xyz)) return; // don't execute graphics_q() on marching-cubes halo
)+"#else"+R( // do not use local memory
	if(n>=(uxx)(def_Nx-1u)*(uxx)(def_Ny-1u)*(uxx)(def_Nz-1u)) return;
	const uint3 xyz = coordinates_mc(n);
	if(is_halo_mc(xyz)) return; // don't execute graphics_q() on marching-cubes halo
)+"#endif"+R( // do not use local memory
	const float3 p = position(xyz);
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
	float3 uj[32];
)+"#if LSQ>0u"+R( // use local memory
	{ // load 32-cell stencil from cache
		const uint xm=xyz_local.x                    , x0=xm+               1u  , xp=xm+ 2u                   , xq=xm+ 3u                   ;
		const uint ym=xyz_local.y*          (LSQ+3u) , y0=ym+          (LSQ+3u) , yp=ym+(2u*         (LSQ+3u)), yq=ym+(3u*         (LSQ+3u));
		const uint zm=xyz_local.z*((LSQ+3u)*(LSQ+3u)), z0=zm+((LSQ+3u)*(LSQ+3u)), zp=zm+(2u*(LSQ+3u)*(LSQ+3u)), zq=zm+(3u*(LSQ+3u)*(LSQ+3u));
		uj[ 0] = u_cache[x0+y0+z0]; // 000 // cube stencil
		uj[ 1] = u_cache[xp+y0+z0]; // +00
		uj[ 2] = u_cache[xp+y0+zp]; // +0+
		uj[ 3] = u_cache[x0+y0+zp]; // 00+
		uj[ 4] = u_cache[x0+yp+z0]; // 0+0
		uj[ 5] = u_cache[xp+yp+z0]; // ++0
		uj[ 6] = u_cache[xp+yp+zp]; // +++
		uj[ 7] = u_cache[x0+yp+zp]; // 0++
		uj[ 8] = u_cache[xm+y0+z0]; // -00 // central difference stencil on each cube corner point
		uj[ 9] = u_cache[x0+ym+z0]; // 0-0
		uj[10] = u_cache[x0+y0+zm]; // 00-
		uj[11] = u_cache[xq+y0+z0]; // #00
		uj[12] = u_cache[xp+ym+z0]; // +-0
		uj[13] = u_cache[xp+y0+zm]; // +0-
		uj[14] = u_cache[xq+y0+zp]; // #0+
		uj[15] = u_cache[xp+ym+zp]; // +-+
		uj[16] = u_cache[xp+y0+zq]; // +0#
		uj[17] = u_cache[xm+y0+zp]; // -0+
		uj[18] = u_cache[x0+ym+zp]; // 0-+
		uj[19] = u_cache[x0+y0+zq]; // 00#
		uj[20] = u_cache[xm+yp+z0]; // -+0
		uj[21] = u_cache[x0+yq+z0]; // 0#0
		uj[22] = u_cache[x0+yp+zm]; // 0+-
		uj[23] = u_cache[xq+yp+z0]; // #+0
		uj[24] = u_cache[xp+yq+z0]; // +#0
		uj[25] = u_cache[xp+yp+zm]; // ++-
		uj[26] = u_cache[xq+yp+zp]; // #++
		uj[27] = u_cache[xp+yq+zp]; // +#+
		uj[28] = u_cache[xp+yp+zq]; // ++#
		uj[29] = u_cache[xm+yp+zp]; // -++
		uj[30] = u_cache[x0+yq+zp]; // 0#+
		uj[31] = u_cache[x0+yp+zq]; // 0+#
	}
	uint j[8];
	calculate_j8(xyz, j);
)+"#else"+R( // do not use local memory
	uxx j[32];
	calculate_j32(xyz, j);
	for(uint i=0u; i<32u; i++) uj[i] = load3(u, j[i]);
)+"#endif"+R( // do not use local memory
)+"#ifdef SURFACE"+R(
	uchar flags_cell = 0u;
	for(uint i=0u; i<8u; i++) flags_cell |= flags[j[i]];
	if(flags_cell&(TYPE_I|TYPE_G)) return;
)+"#endif"+R( // SURFACE
	float v[8]; // don't load any velocity twice from global memory
	v[0] = calculate_Q_cached(uj[ 1], uj[ 8], uj[ 4], uj[ 9], uj[ 3], uj[10]);
	v[1] = calculate_Q_cached(uj[11], uj[ 0], uj[ 5], uj[12], uj[ 2], uj[13]);
	v[2] = calculate_Q_cached(uj[14], uj[ 3], uj[ 6], uj[15], uj[16], uj[ 1]);
	v[3] = calculate_Q_cached(uj[ 2], uj[17], uj[ 7], uj[18], uj[19], uj[ 0]);
	v[4] = calculate_Q_cached(uj[ 5], uj[20], uj[21], uj[ 0], uj[ 7], uj[22]);
	v[5] = calculate_Q_cached(uj[23], uj[ 4], uj[24], uj[ 1], uj[ 6], uj[25]);
	v[6] = calculate_Q_cached(uj[26], uj[ 7], uj[27], uj[ 2], uj[28], uj[ 5]);
	v[7] = calculate_Q_cached(uj[ 6], uj[29], uj[30], uj[ 3], uj[31], uj[ 4]);
	float3 triangles[15]; // maximum of 5 triangles with 3 vertices each
	const uint tn = marching_cubes(v, def_scale_Q_min, triangles); // run marching cubes algorithm
	if(tn==0u) return;
	for(uint i=0u; i<tn; i++) {
		const float3 p0 = triangles[3u*i   ]; // triangle coordinates in [0,1] (local cell)
		const float3 p1 = triangles[3u*i+1u];
		const float3 p2 = triangles[3u*i+2u];
		const float3 normal = cross(p1-p0, p2-p0); // no normalize needed for shading()
		int c0=0, c1=0, c2=0;
		switch(field_mode) {
			case 0: // coloring by velocity
				c0 = shading(colorscale_rainbow(def_scale_u*length(trilinear3(p0, uj))), p+p0, normal, camera_cache);
				c1 = shading(colorscale_rainbow(def_scale_u*length(trilinear3(p1, uj))), p+p1, normal, camera_cache);
				c2 = shading(colorscale_rainbow(def_scale_u*length(trilinear3(p2, uj))), p+p2, normal, camera_cache);
				break;
			case 1: // coloring by density
				for(uint i=0u; i<8u; i++) v[i] = rho[j[i]];
				c0 = shading(colorscale_twocolor(0.5f+def_scale_rho*(trilinear(p0, v)-1.0f)), p+p0, normal, camera_cache);
				c1 = shading(colorscale_twocolor(0.5f+def_scale_rho*(trilinear(p1, v)-1.0f)), p+p1, normal, camera_cache);
				c2 = shading(colorscale_twocolor(0.5f+def_scale_rho*(trilinear(p2, v)-1.0f)), p+p2, normal, camera_cache);
				break;
)+"#ifdef TEMPERATURE"+R(
			case 2: // coloring by temperature
				for(uint i=0u; i<8u; i++) v[i] = T[j[i]];
				c0 = shading(colorscale_iron(0.5f+def_scale_T*(trilinear(p0, v)-def_T_avg)), p+p0, normal, camera_cache);
				c1 = shading(colorscale_iron(0.5f+def_scale_T*(trilinear(p1, v)-def_T_avg)), p+p1, normal, camera_cache);
				c2 = shading(colorscale_iron(0.5f+def_scale_T*(trilinear(p2, v)-def_T_avg)), p+p2, normal, camera_cache);
				break;
)+"#endif"+R( // TEMPERATURE
		}
		draw_triangle_interpolated(p+p0, p+p1, p+p2, c0, c1, c2, camera_cache, bitmap, zbuffer); // draw triangle with interpolated colors
	}
}

)+"#ifdef SURFACE"+R(
)+R(kernel void graphics_rasterize_phi(const global float* camera, global int* bitmap, global int* zbuffer, const global float* phi) { // marching cubes
	const uxx n = get_global_id(0);
)+"#if LSP>0u"+R( // use local memory
	const uxx n_global = n/(LSP*LSP*LSP);
	const uint n_local = (uint)(n%(LSP*LSP*LSP));
	const uint t_global = (uint)(n_global%(uxx)(((def_Nx+LSP-2u)/LSP)*((def_Ny+LSP-2u)/LSP))); // -1u for coordinates_mc, +LSP-1u for always rounding up
	const uint3 xyz_global = (uint3)(t_global%((def_Nx+LSP-2u)/LSP), t_global/((def_Nx+LSP-2u)/LSP), (uint)(n_global/(uxx)(((def_Nx+LSP-2u)/LSP)*((def_Ny+LSP-2u)/LSP)))); // n = x+(y+z*Ny)*Nx
	const uint t_local = n_local%(LSP*LSP);
	const uint3 xyz_local = (uint3)(t_local%LSP, t_local/LSP, n_local/(LSP*LSP)); // n = x+(y+z*Ny)*Nx
	const uint3 xyz = LSP*xyz_global+xyz_local;
	local float phi_cache[(LSP+1u)*(LSP+1u)*(LSP+1u)]; // for LSP==4: 4x4x4 cells with [0,+1] halo = 5x5x5 cells (500 Byte) to load in cache
	const uint loads_per_thread = ((LSP+1u)*(LSP+1u)*(LSP+1u)+LSP*LSP*LSP-1u)/(LSP*LSP*LSP); // number of grid cells each thread has to load (for LSP==4: 2, for LSP==8: 2)
	for(uint c=0u; c<loads_per_thread; c++) {
		const uint n_local_c = c*(LSP*LSP*LSP)+n_local; // SoA (>2x faster on GPUs)
		if(n_local_c<(LSP+1u)*(LSP+1u)*(LSP+1u)) {
			const uint t_local_c = n_local_c%((LSP+1u)*(LSP+1u));
			const uint3 xyz_local_c = (uint3)(t_local_c%(LSP+1u), t_local_c/(LSP+1u), n_local_c/((LSP+1u)*(LSP+1u))); // n = x+(y+z*Ny)*Nx
			const uint3 xyz_global_c = (LSP*xyz_global+xyz_local_c)%(uint3)(def_Nx, def_Ny, def_Nz); // apply periodic boundaries
			phi_cache[n_local_c] = phi[index(xyz_global_c)];
		}
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(xyz.x>=def_Nx-1u||xyz.y>=def_Ny-1u||xyz.z>=def_Nz-1u||is_halo_mc(xyz)) return; // don't execute graphics_rasterize_phi() on marching-cubes halo
)+"#else"+R( // do not use local memory
	if(n>=(uxx)(def_Nx-1u)*(uxx)(def_Ny-1u)*(uxx)(def_Nz-1u)) return;
	const uint3 xyz = coordinates_mc(n);
	if(is_halo_mc(xyz)) return; // don't execute graphics_rasterize_phi() on marching-cubes halo
)+"#endif"+R( // do not use local memory
	const float3 p = position(xyz);
	float camera_cache[15]; // cache camera parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	if(!is_in_camera_frustum(p, camera_cache)) return; // skip loading LBM data if grid cell is not visible
	float v[8];
)+"#if LSP>0u"+R( // use local memory
	{ // load 8-cell stencil from cache
		const uint x0=xyz_local.x                    , xp=x0+               1u  ;
		const uint y0=xyz_local.y*          (LSP+1u) , yp=y0+          (LSP+1u) ;
		const uint z0=xyz_local.z*((LSP+1u)*(LSP+1u)), zp=z0+((LSP+1u)*(LSP+1u));
		v[0] = phi_cache[x0+y0+z0]; // 000 // cube stencil
		v[1] = phi_cache[xp+y0+z0]; // +00
		v[2] = phi_cache[xp+y0+zp]; // +0+
		v[3] = phi_cache[x0+y0+zp]; // 00+
		v[4] = phi_cache[x0+yp+z0]; // 0+0
		v[5] = phi_cache[xp+yp+z0]; // ++0
		v[6] = phi_cache[xp+yp+zp]; // +++
		v[7] = phi_cache[x0+yp+zp]; // 0++
	}
)+"#else"+R( // do not use local memory
	uxx j[8];
	calculate_j8(xyz, j);
	for(uint i=0u; i<8u; i++) v[i] = phi[j[i]];
)+"#endif"+R( // do not use local memory
	float3 triangles[15]; // maximum of 5 triangles with 3 vertices each
	const uint tn = marching_cubes(v, 0.51f, triangles); // run marching cubes algorithm, isovalue slightly larger than 0.5f to fix z-fighting with graphics_flags_mc()
	if(tn==0u) return;
	for(uint i=0u; i<tn; i++) {
		const float3 p0 = triangles[3u*i   ];
		const float3 p1 = triangles[3u*i+1u];
		const float3 p2 = triangles[3u*i+2u];
		const float3 normal = cross(p1-p0, p2-p0); // no normalize needed for shading()
		const int c0 = shading(0x379BFF, p+p0, normal, camera_cache);
		const int c1 = shading(0x379BFF, p+p1, normal, camera_cache);
		const int c2 = shading(0x379BFF, p+p2, normal, camera_cache);
		draw_triangle_interpolated(p+p0, p+p1, p+p2, c0, c1, c2, camera_cache, bitmap, zbuffer);
		//const int c = shading(0x379BFF, p+(p0+p1+p2)/3.0f, normal, camera_cache);
		//draw_line(p+p0, p+p1, c, camera_cache, bitmap, zbuffer); // wireframe rendering
		//draw_line(p+p0, p+p2, c, camera_cache, bitmap, zbuffer);
		//draw_line(p+p1, p+p2, c, camera_cache, bitmap, zbuffer);
	}
}

)+R(int raytrace_phi_next_ray(const ray reflection, const ray transmission, const float reflectivity, const float transmissivity, const global float* phi, const global uchar* flags, const global int* skybox) {
	int color_reflect=0, color_transmit=0;
	ray reflection_next, transmission_next;
	float reflection_reflectivity, reflection_transmissivity, transmission_reflectivity, transmission_transmissivity;
	if(raytrace_phi(reflection, &reflection_next, &transmission_next, &reflection_reflectivity, &reflection_transmissivity, phi, flags, skybox, def_Nx, def_Ny, def_Nz)) {
		color_reflect = last_ray(reflection_next, transmission_next, reflection_reflectivity, reflection_transmissivity, skybox);
	} else {
		color_reflect = skybox_color(reflection, skybox);
	}
	if(raytrace_phi(transmission, &reflection_next, &transmission_next, &transmission_reflectivity, &transmission_transmissivity, phi, flags, skybox, def_Nx, def_Ny, def_Nz)) {
		color_transmit = last_ray(reflection_next, transmission_next, transmission_reflectivity, transmission_transmissivity, skybox);
	} else {
		color_transmit = skybox_color(transmission, skybox);
	}
	return color_mix(color_reflect, color_mix(color_transmit, def_absorption_color, transmissivity), reflectivity);
}
)+R(int raytrace_phi_next_ray_mirror(const ray reflection, const global float* phi, const global uchar* flags, const global int* skybox) {
	int color_reflect = 0;
	ray reflection_next;
	if(raytrace_phi_mirror(reflection, &reflection_next, phi, flags, skybox, def_Nx, def_Ny, def_Nz)) {
		color_reflect = skybox_color(reflection_next, skybox);
	} else {
		color_reflect = skybox_color(reflection, skybox);
	}
	return color_reflect;
}

)+R(kernel void graphics_raytrace_phi(const global float* camera, global int* bitmap, const global int* skybox, const global float* phi, const global uchar* flags) { // marching cubes
	const uint gid = get_global_id(0); // workgroup size alignment is critical
	const uint lid = get_local_id(0); // make workgropus not horizontal stripes of pixels, but 8x8 rectangular (close to square) tiles
	const uint lsi = get_local_size(0); // (50% performance boost due to more coalesced memory access)
	const uint tile_width=8u, tile_height=lsi/tile_width, tiles_x=def_screen_width/tile_width;
	const int lx=lid%tile_width, ly=lid/tile_width;
	const int tx=(gid/lsi)%tiles_x, ty=(gid/lsi)/tiles_x;
	const int x=tx*tile_width+lx, y=ty*tile_height+ly;
	const uint n = x+y*def_screen_width;
	float camera_cache[15]; // cache parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	ray camray = get_camray(x, y, camera_cache);
	const float distance = intersect_cuboid(camray, (float3)(0.0f, 0.0f, 0.0f), (float)def_Nx, (float)def_Ny, (float)def_Nz);
	camray.origin = camray.origin+fmax(distance+0.005f, 0.005f)*camray.direction;
	ray reflection, transmission; // reflection and transmission
	float reflectivity, transmissivity;
	int pixelcolor = 0;
	if(raytrace_phi(camray, &reflection, &transmission, &reflectivity, &transmissivity, phi, flags, skybox, def_Nx, def_Ny, def_Nz)) {
		pixelcolor = last_ray(reflection, transmission, reflectivity, transmissivity, skybox); // 1 ray pass
		//pixelcolor = raytrace_phi_next_ray(reflection, transmission, reflectivity, transmissivity, phi, flags, skybox); // 2 ray passes
	} else {
		pixelcolor = skybox_color(camray, skybox);
	}
	//if(raytrace_phi_mirror(camray, &reflection, phi, flags, skybox, def_Nx, def_Ny, def_Nz)) { // reflection only
	//	//pixelcolor = skybox_color(reflection, skybox); // 1 ray pass
	//	pixelcolor = raytrace_phi_next_ray_mirror(reflection, phi, flags, skybox); // 2 ray passes
	//} else {
	//	pixelcolor = skybox_color(camray, skybox);
	//}
	bitmap[n] = pixelcolor; // no zbuffer required
}
)+"#endif"+R( // SURFACE

)+"#ifdef PARTICLES"+R(
)+R(kernel void graphics_particles(const global float* camera, global int* bitmap, global int* zbuffer, const global float* particles) {
	const uxx n = get_global_id(0);
	if(n>=(uxx)def_particles_N) return;
	const float3 p = (float3)(particles[n]-def_domain_offset_x, particles[def_particles_N+(ulong)n]-def_domain_offset_y, particles[2ul*def_particles_N+(ulong)n]-def_domain_offset_z);
	if(def_Dx*def_Dy*def_Dz>1u&&!position_is_in_domain_excluding_halo(p)) return;
	float camera_cache[15]; // cache parameters in case the kernel draws more than one shape
	for(uint i=0u; i<15u; i++) camera_cache[i] = camera[i];
	const int c = COLOR_P; // coloring scheme
	draw_point(p, c, camera_cache, bitmap, zbuffer);
	//draw_circle(p, 0.5f, c, camera_cache, bitmap, zbuffer);
}
)+"#endif"+R( // PARTICLES
)+"#endif"+R( // GRAPHICS



);} // ############################################################### end of OpenCL C code #####################################################################

// ★ 15.09.2026 Klemmen Stufe 1 P1a (KLEMMEN-STUFE1-PLAN.md §2a, §4): EINZIGE Quelle der Positivitaets-Defines. lbm.cpp (Emission) und
// das Scratch-Gate (gen_main datei ... pos*) rufen dieselbe Funktion -- Gate und Lauf koennen nicht auseinanderlaufen.
// modus 0 = leer (Kernelquelle zeichengleich), 1 = POSITIV (Messarm: s gerechnet und gezaehlt, Felder bitgleich), 2 = zusaetzlich
// POSITIV_ANWENDEN; facette 1 = K0 (Facettenzellen) in Modus 2 eingeschlossen (Vorgabe 0 = ausgenommen, Entscheidung E2).
// tau_i = halbe ULP des FP16S-Halbworts bei f = 0: Rechenform f - w_i = -w_i, gespeichert -w_i*2^15 (i = 0: 10 922,7 -> ULP 8;
// Achsen 1 820,4 -> ULP 1; Diagonalen 910,2 -> ULP 1/2), vstore_half_rte rundet zur naechsten Stufe -> nach dem LADEN f >= 0 (gespeichert wird f - w_i). Pruefbefund P1a: weil -w_i 2^15 nicht auf dem Raster liegt,
// liefert schon tau = 0 dieselben Minima -- tau ist vorsichtig, [272] zaehlt f* < tau, nicht f* < 0; [285] sieht deshalb weniger.
// FP32: tau = 0 (f >= 0 nur bis auf die float-Rundung, ~1e-8). Haken 2 (Negativtest): tau_i = 1,2*w_i, jede Zelle ist Kandidat UND machtlos (B_0 < 1,2 w_0 fuer rho < 1,2). Als AUSDRUECKE emittiert (Plan §2a, nicht to_string).
unsigned positiv_haken_periode() { return 1009u; } // ★ P1b: Haken-1-Zellen n mod P == 0 (Primzahl, keine Gitterresonanz); EINE Quelle fuer Kernel-Define und Host-Soll
// ★ P1b Nachtrag: sicheres_gitter = groesstes Gitter, an dem Atomics in (fast) jeder Zelle je Schritt nachweislich liefen (Kugel 16 mm,
// 6 104 700 Zellen, Absturzsperre lbm.cpp). Die Zellzaehler des Messarms laufen nur fuer n mod def_pos_sub == 0 mit
// def_pos_sub = kleinste Primzahl >= ceil(N / sicheres_gitter), die Nx und Ny nicht teilt (positiv_stichprobe): je Zaehlschritt hoechstens so viele zaehlende Zellen wie am belegt sicheren Gitter -- automatisch, kein Handwert.
unsigned long long positiv_sicheres_gitter() { return 6104700ull; } // EINE Quelle: Absturzsperre (lbm.cpp) und Stichprobe (Pruefbefund P1b NIEDRIG 4)
// Pruefbefund P1b NIEDRIG 3: die Periode ist die kleinste PRIMZAHL >= ceil(N/g), die weder Nx noch Ny teilt -- sonst zaehlt n%p nur jede p-te
// x-Ebene (Fernfeld 8 mm: 5 teilt Nx = 385). N/p <= g bleibt erhalten.
unsigned positiv_stichprobe(const unsigned long long N, const unsigned Nx, const unsigned Ny) {
	const unsigned long long g = positiv_sicheres_gitter(), c = (N+g-1ull)/g;
	if(c<=1ull) return 1u;
	for(unsigned long long p=c; ; p++) {
		bool prim = p>=2ull; for(unsigned long long d=2ull; d*d<=p&&prim; d++) if(p%d==0ull) prim = false;
		if(prim&&Nx%p!=0ull&&Ny%p!=0ull) return (unsigned)p;
	}
}
string positiv_defines(const unsigned modus, const unsigned haken, const unsigned facette, const bool fp16s, const unsigned long long N, const unsigned Nx, const unsigned Ny) { // unsigned statt uint: kernel.hpp definiert uint fuer die Syntaxfaerbung leer
	if(modus==0u) return "";
	string s = "\n	#define POSITIV";
	if(modus>=2u) s += "\n	#define POSITIV_ANWENDEN";
	if(facette>0u) s += "\n	#define POSITIV_FACETTE";
	s += "\n	#define def_pos_sub "+std::to_string(positiv_stichprobe(N, Nx, Ny))+"u"; // Stichprobenperiode der Zellzaehler
	if(haken==2u) s += "\n	#define def_pos_t0 (1.2f*def_w0)\n	#define def_pos_ts (1.2f*def_ws)\n	#define def_pos_te (1.2f*def_we)";
	else if(fp16s) s += "\n	#define def_pos_t0 (4.0f/32768.0f)\n	#define def_pos_ts (0.5f/32768.0f)\n	#define def_pos_te (0.25f/32768.0f)";
	else s += "\n	#define def_pos_t0 0.0f\n	#define def_pos_ts 0.0f\n	#define def_pos_te 0.0f";
	s += "\n	#define def_pos_g0 (def_pos_t0-def_w0)\n	#define def_pos_gs (def_pos_ts-def_ws)\n	#define def_pos_ge (def_pos_te-def_we)"; // Kandidat: fhn[i] < tau_i - w_i
	if(haken==1u||haken==3u) s += "\n	#define POSITIV_HAKEN1\n	#define def_pos_hP "+std::to_string(positiv_haken_periode())+"u"; // Periode der Hakenzellen (Host-Soll liest dieselbe Funktion)
	if(haken==3u) s += "\n	#define POSITIV_HAKEN3"; // H3 = H1-Stoerung + Klassenluecke: ohne natuerliche Kandidaten (Kugel 40 mm: 0) feuerte der Negativtest sonst nie (P1b-CPU 15.09.)
	return s;
}
