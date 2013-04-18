#pragma once

#include <windows.h>
#include <string>

using namespace std;

static LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam );


// =======================================================================================
//                                      Context
// =======================================================================================

///---------------------------------------------------------------------------------------
/// \brief	Window context for DirectX
///        
/// # Context
/// 
/// 17-4-2013 Jarl Larsson
///---------------------------------------------------------------------------------------

class Context
{
public:
	Context(HINSTANCE p_hInstance, const string& p_title,
		int p_width, int p_height);
	virtual ~Context();
	HWND getWindowHandle();
	static Context* getInstance();

	void close();

	void resize(int p_w, int p_h);

	///-----------------------------------------------------------------------------------
	/// Change the window title string
	/// \param p_title
	/// \return void
	///-----------------------------------------------------------------------------------
	void setTitle(const string& p_title);

	///-----------------------------------------------------------------------------------
	/// Update the window with the store title
	/// \param p_appendMsg Optional message string to append to title
	/// \return void
	///-----------------------------------------------------------------------------------
	void updateTitle(const string& p_appendMsg="");

	///-----------------------------------------------------------------------------------
	/// Whether a closedown was requested
	/// \return bool
	///-----------------------------------------------------------------------------------
	bool closeRequested() const;
protected:
private:
	bool m_closeFlag;
	string m_title;
	int m_width;
	int m_height;
	HINSTANCE	m_hInstance;
	HWND		m_hWnd;
	static Context* m_instance;
};