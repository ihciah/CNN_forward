/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include <windows.h>
#include "CnnNet.h"
#include "utils.h"

class Cap_rec{
public:
	void init(char*);
	void rec(char* path, char* result);
private:
	CnnNet net;
}cr;

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
	)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

extern "C" {
	_declspec(dllexport) void _stdcall rec(char* path, char* result) { return cr.rec(path, result); }
}
extern "C" {
	_declspec(dllexport) void _stdcall init(char* path) { return cr.init(path); }
}
