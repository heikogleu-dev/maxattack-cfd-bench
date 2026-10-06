#include "info.hpp"
#include "lbm.hpp"

Info info;

void Info::append(const ulong steps, const ulong total_steps, const ulong t) {
	if(total_steps==max_ulong) { // total_steps is not provided/used
		this->steps = steps; // has to be executed before info.print_initialize()
		this->steps_last = t; // reset last step count if multiple run() commands are executed consecutively
		this->runtime_total_last = this->runtime_total; // reset last runtime if multiple run() commands are executed consecutively
		this->runtime_total = clock.stop();
	} else { // total_steps has been specified
		this->steps = total_steps; // has to be executed before info.print_initialize()
	}
}
void Info::update(const double dt) {
	this->runtime_lbm_timestep_last = dt; // exact dt
	this->runtime_lbm_timestep_smooth = (dt+0.3)/(0.3/runtime_lbm_timestep_smooth+1.0); // smoothed dt
	this->runtime_lbm += dt; // skip first step since it is likely slower than average
	this->runtime_total = clock.stop();
}
// ★ 03.10.2026 (Heiko: Leistungsanzeige je GPU ehrlich). Im fahrzeug_dd-Fall zeigte die Laufzeile info.lbm = das zuletzt
// per run(0) initialisierte LBM, also das FERNfeld (N_fern, t_fern), waehrend info.update() nur aus lbm_f.run() kam, also mit
// der NAHfeld-Schrittzeit: MLUPs = N_fern / t_nah. Dazu setzte run_async() in jedem Aussenschritt die Zaehler zurueck.
// lauf_binden() setzt Gitter, Schrittzaehler und Uhren in EINEM Zug unter dem Druck-Lock -- die Anzeige sieht nie einen
// halb gesetzten Stand. Danach misst die Zeile genau EINE Domaene mit IHRER Schrittzeit.
void Info::lauf_binden(LBM* lbm, const ulong total_steps) {
	allow_printing.lock();
	this->lbm = lbm;
	this->steps = total_steps;
	this->steps_last = lbm->get_t();
	this->runtime_lbm = 0.0;
	this->runtime_lbm_timestep_smooth = 1.0;
	this->runtime_lbm_timestep_last = 1.0;
	this->runtime_total = 0.0;
	this->runtime_total_last = 0.0;
	clock.start();
	allow_printing.unlock();
}
double Info::time() const { // returns either elapsed time or remaining time
	if(lbm==nullptr) return 0.0;
	return steps==max_ulong ? runtime_total : ((double)steps/(double)max(lbm->get_t()-steps_last, 1ull)-1.0)*(runtime_total-runtime_total_last); // time estimation on average so far
	//return steps==max_ulong ? runtime_lbm : ((double)steps-(double)(lbm->get_t()-steps_last))*runtime_lbm_timestep_smooth; // instantaneous time estimation
}
void Info::print_logo() const {
	const int a=color_light_blue, b=color_orange, c=color_pink;
	print(".-----------------------------------------------------------------------------.\n");
	print("|                      "); print(  " ______________  ", a);                  print(" ______________ ", b); print("                      |\n");
	print("|                       "); print( "\\   ________  | ", a);                  print("|  ________   /", b); print("                       |\n");
	print("|                        "); print("\\  \\       | | ", a);                  print("| |       /  /", b); print("                        |\n");
	print("|                         "); print("\\  \\      | | ", a);                  print("| |      /  /", b); print("                         |\n");
	print("|                          "); print("\\  \\     | | ", a);                  print("| |     /  /", b); print("                          |\n");
	print("|                           "); print("\\  \\_.-\"  | ", a);                print("|  \"-._/  /", b); print("                           |\n");
	print("|                            "); print("\\    _.-\" ", a);  print("_ ", c);  print("\"-._    /", b); print("                            |\n");
	print("|                             "); print("\\.-\" ", a); print("_.-\" \"-._ ", c); print("\"-./", b); print("                             |\n");
	print("|                              ");                 print(" .-\"  .-\"-.  \"-. ", c);               print("                              |\n");
	print("|                               ");                 print("\\  v\"     \"v  /", c);               print("                               |\n");
	print("|                                ");                 print("\\  \\     /  /", c);                print("                                |\n");
	print("|                                 ");                 print("\\  \\   /  /", c);                print("                                 |\n");
	print("|                                  ");                 print("\\  \\ /  /", c);                print("                                  |\n");
	print("|                                   ");                 print("\\  '  /", c);                 print("                                   |\n");
	print("|                                    ");                 print("\\   /", c);                 print("                                    |\n");
	print("|                                     ");                 print("\\ /", c);                 print("                 MaxAttack CFD Bench |\n");
	print("|                                      ");                 print( "'", c);                 print("     Copyright (c) Dr. Moritz Lehmann |\n");
	print("|                                          modified from FluidX3D Version 3.7 |\n");
	print("|                                            github.com/ProjectPhysX/FluidX3D |\n");
	print("|-----------------------------------------------------------------------------|\n");
}
void Info::print_initialize(LBM* lbm) {
	info.allow_printing.lock(); // disable print_update() until print_initialize() has finished
	this->lbm = lbm;
#if defined(SRT)
	collision = "SRT";
#elif defined(TRT)
	collision = "TRT";
#endif // TRT
#if defined(FP16S)
	collision += " (FP32/FP16S)";
#elif defined(FP16C)
	collision += " (FP32/FP16C)";
#else // FP32
	collision += " (FP32/FP32)";
#endif // FP32
	bool all_domains_use_ram = true; // reset cpu/gpu_mem_required to get valid values for consecutive simulations
	for(uint d=0u; d<lbm->get_D(); d++) {
		all_domains_use_ram = all_domains_use_ram&&lbm->lbm_domain[d]->get_device().info.uses_ram;
	}
	if(all_domains_use_ram) {
		cpu_mem_required = lbm->get_D()*lbm->lbm_domain[0]->get_device().info.memory_used;
		gpu_mem_required = 0u;
	} else {
		// ★ 29.08.: dieselbe Wurzel wie der VRAM-Befund -- bytes_per_cell_host() rechnet
		// FORCE_FIELD mit 12 B/Zelle ueber das VOLLE Gitter, obwohl F host- wie geraeteseitig
		// nur ueber die Bounding-Box liegt (lbm.cpp:344-347). Beim 4-mm-Fahrzeug ueberzeichnete
		// die Anzeige den Hostbedarf dadurch um rund 3.989 MB. Jetzt derselbe Ausdruck wie dort.
		{	const LBM_Domain* d0 = lbm->lbm_domain[0];
			const ulong F_N = (ulong)d0->fbnx*(ulong)d0->fbny*(ulong)d0->fbnz;
			ulong b = lbm->get_N()*(ulong)bytes_per_cell_host();
#ifdef FORCE_FIELD
			b -= 12ull*(lbm->get_N() - (ulong)lbm->get_D()*F_N); // F liegt nur ueber die BBox
#endif // FORCE_FIELD
			if(d0->rho_rand_on) b -= (ulong)sizeof(rhoxx)*(lbm->get_N()-(d0->rr_N+1ull)); // ★ 15.09. RHO_RAND C2c: rho-Hostspiegel nur R1
			cpu_mem_required = (uint)(b/1048576ull); }
		gpu_mem_required = lbm->lbm_domain[0]->get_device().info.memory_used;
	}
	const float Re = lbm->get_Re_max();
	println("|-----------------.-----------------------------------------------------------|");
	println("| Grid Resolution | "+alignr(57u, to_string(lbm->get_Nx())+" x "+to_string(lbm->get_Ny())+" x "+to_string(lbm->get_Nz())+" = "+to_string(lbm->get_N()))+" |");
	println("| Grid Domains    | "+alignr(57u, to_string(lbm->get_Dx())+" x "+to_string(lbm->get_Dy())+" x "+to_string(lbm->get_Dz())+" = "+to_string(lbm->get_D()))+" |");
	println("| LBM Type        | "+alignr(57u, /***************/ "D"+to_string(lbm->get_velocity_set()==9?2:3)+"Q"+to_string(lbm->get_velocity_set())+" "+collision)+" |");
	println("| Memory Usage    | "+alignr(54u, /*******/ "CPU "+to_string(cpu_mem_required)+" MB, GPU "+to_string(lbm->get_D())+"x "+to_string(gpu_mem_required))+" MB |");
	println("| Max Alloc Size  | "+alignr(54u, /*************/ (uint)(lbm->get_N()/(ulong)lbm->get_D()*(ulong)(lbm->get_velocity_set()*sizeof(fpxx))/1048576ull))+" MB |");
	println("| Time Steps      | "+alignr(57u, /***************************************************************/ (steps==max_ulong ? "infinite" : to_string(steps)))+" |");
	println("| Kin. Viscosity  | "+alignr(57u, /*************************************************************************************/ to_string(lbm->get_nu(), 8u))+" |");
	println("| Relaxation Time | "+alignr(57u, /************************************************************************************/ to_string(lbm->get_tau(), 8u))+" |");
	println("| Reynolds Number | "+alignr(57u, /******************************************/ "Re < "+string(Re>=100.0f ? to_string(to_uint(Re)) : to_string(Re, 6u)))+" |");
#ifdef VOLUME_FORCE
	println("| Volume Force    | "+alignr(57u, alignr(15u, to_string(lbm->get_fx(), 8u))+","+alignr(15u, to_string(lbm->get_fy(), 8u))+","+alignr(15u, to_string(lbm->get_fz(), 8u)))+" |");
#endif // VOLUME_FORCE
#ifdef SURFACE
	println("| Surface Tension | "+alignr(57u, /**********************************************************************************/ to_string(lbm->get_sigma(), 8u))+" |");
#endif // SURFACE
#ifdef TEMPERATURE
	println("| Thermal Diff.   | "+alignr(57u, /**********************************************************************************/ to_string(lbm->get_alpha(), 8u))+" |");
	println("| Thermal Exp.    | "+alignr(57u, /***********************************************************************************/ to_string(lbm->get_beta(), 8u))+" |");
#endif // TEMPERATURE
#ifndef INTERACTIVE_GRAPHICS_ASCII
	println("|---------.-------'-----.-----------.-------------------.---------------------|");
	println("| MLUPs   | Bandwidth   | Steps/s   | Current Step      | "+string(steps==max_ulong?"Elapsed Time  ":"Time Remaining")+"      |");
#else // INTERACTIVE_GRAPHICS_ASCII
	println("'-----------------'-----------------------------------------------------------'");
#endif // INTERACTIVE_GRAPHICS_ASCII
	clock.start();
	info.allow_printing.unlock();
}
void Info::print_update() const {
	if(lbm==nullptr) return;
	info.allow_printing.lock();
	reprint(
		"|"+alignr(8, to_uint((double)lbm->get_N()*1E-6/runtime_lbm_timestep_smooth))+" |"+ // MLUPs
		alignr(7, to_uint((double)lbm->get_N()*(double)bpz_konv(*lbm)*1E-9/runtime_lbm_timestep_smooth))+" GB/s |"+ // memory bandwidth (★ 03.10.2026 bpz_konv = Upstream-Konvention, keine Messung)
		alignr(10, to_uint(1.0/runtime_lbm_timestep_smooth))+" | "+ // steps/s
		(steps==max_ulong ? alignr(17, lbm->get_t()) : alignr(12, lbm->get_t())+" "+print_percentage((float)(lbm->get_t()-steps_last)/(float)steps))+" | "+ // current step
		alignr(19, print_time(time()))+" |" // either elapsed time or remaining time
	);
#ifdef GRAPHICS
	if(key_G) { // print camera settings
		const string camera_position = "float3("+alignr(9u, to_string(camera.pos.x/(float)lbm->get_Nx(), 6u))+"f*(float)Nx, "+alignr(9u, to_string(camera.pos.y/(float)lbm->get_Ny(), 6u))+"f*(float)Ny, "+alignr(9u, to_string(camera.pos.z/(float)lbm->get_Nz(), 6u))+"f*(float)Nz)";
		const string camera_rx_ry_fov = alignr(6u, to_string(degrees(camera.rx)-90.0, 1u))+"f, "+alignr(5u, to_string(180.0-degrees(camera.ry), 1u))+"f, "+alignr(5u, to_string(camera.fov, 1u))+"f";
		const string camera_zoom = alignr(8u, to_string(camera.zoom*(float)fmax(fmax(lbm->get_Nx(), lbm->get_Ny()), lbm->get_Nz())/(float)min(camera.width, camera.height), 6u))+"f";
		if(camera.free) println("\rlbm.graphics.set_camera_free("+camera_position+", "+camera_rx_ry_fov+");");
		else println("\rlbm.graphics.set_camera_centered("+camera_rx_ry_fov+", "+camera_zoom+");          ");
		key_G = false;
	}
#endif // GRAPHICS
	info.allow_printing.unlock();
}
void Info::print_finalize() {
	lbm = nullptr;
	println("\n|---------'-------------'-----------'-------------------'---------------------|");
}