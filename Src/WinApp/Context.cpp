#include "Context.h"
#include "ContextException.h"

Context::Context( HINSTANCE p_hInstance, const string& p_title, 
				 int p_width, int p_height )
{
	m_hInstance = p_hInstance; 
	m_title = p_title;

	// Register class
	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX); 
	wcex.style          = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc    = WndProc; // Callback function
	wcex.cbClsExtra     = 0;
	wcex.cbWndExtra     = 0;
	wcex.hInstance      = m_hInstance;
	wcex.hIcon          = 0;
	wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
	wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
	wcex.lpszMenuName   = NULL;
	wcex.lpszClassName  = m_title.c_str();
	wcex.hIconSm        = 0;

	if( !RegisterClassEx(&wcex) )
	{
		throw ContextException("Could not register Context class",
			__FILE__,__FUNCTION__,__LINE__);
	}

	// Create the window
	RECT rc = { 0, 0, p_width, p_height};
	AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );
	if(!(m_hWnd = CreateWindow(
		m_title.c_str(),
		m_title.c_str(),
		WS_OVERLAPPEDWINDOW,
		0,0,
		rc.right - rc.left, rc.bottom - rc.top,
		NULL,NULL,
		m_hInstance,
		NULL)))
	{
		throw ContextException("Could not create window",
			__FILE__,__FUNCTION__,__LINE__);
	}

	ShowWindow( m_hWnd, true );
	ShowCursor(true);
}

Context::~Context()
{
	DestroyWindow(m_hWnd);
}

void Context::setTitle( const string& p_title )
{
	m_title=p_title;
}

void Context::updateTitle( const string& p_appendMsg )
{
	SetWindowText(m_hWnd, (m_title+p_appendMsg).c_str());
}


HWND Context::getWindowHandle()
{
	return m_hWnd;
}

void Context::resize( int p_w, int p_h )
{
	m_width = p_w;
	m_height = p_h;
	SetWindowPos( m_hWnd, HWND_TOP, 0, 0, m_width, m_height, SWP_NOMOVE );
}


LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
	PAINTSTRUCT ps;
	HDC hdc;

	switch (message) 
	{
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_KEYDOWN:
		switch(wParam)
		{
		case VK_ESCAPE:
			PostQuitMessage(0);
			break;
		}
		break;

	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	return 0;
}
