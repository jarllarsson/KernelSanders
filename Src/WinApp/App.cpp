#include "App.h"
#include <DebugPrint.h>

#include <Context.h>
#include <ContextException.h>

#include <GraphicsDevice.h>
#include <GraphicsException.h>
#include <BufferFactory.h>

#include "KernelDevice.h"
#include "KernelException.h"

#include <ValueClamp.h>
#include "TempController.h"
#include "ModelImporter.h"
#include "HostSceneManager.h"


#include "OISHelper.h"



const double App::DTCAP=0.5;

App::App( HINSTANCE p_hInstance,MeasurementBin* p_measurer )
{
    int width=600,
        height=400;
    bool windowMode=true;
    // Context
    try
    {
        m_context = new Context(p_hInstance,"Kernel Sanders",width,height);
    }
    catch (ContextException& e)
    {
        DEBUGWARNING((e.what()));
    }	
    
    // Graphics
    try
    {
        m_graphicsDevice = new GraphicsDevice(m_context->getWindowHandle(),width,height,windowMode);
    }
    catch (GraphicsException& e)
    {
        DEBUGWARNING((e.what()));
    }

    // Kernels
    try
    {
        m_kernelDevice = new KernelDevice(m_graphicsDevice->getDevicePointer(),p_measurer);
        m_kernelDevice->registerCanvas(m_graphicsDevice->getInteropCanvasHandle());
    }
    catch (KernelException& e)
    {
        DEBUGWARNING((e.what()));
    }

    // other systems
    fpsUpdateTick=0.0f;
    m_controller = new TempController();
    m_modelImporter = new ModelImporter();
    m_input = new OISHelper();
    m_input->doStartup(m_context->getWindowHandle());
    m_kdDebugBoxInstances=NULL;

    // Finally create and register scene manager
    m_sceneMgr = new HostSceneManager();
    m_kernelDevice->registerSceneMgr(m_sceneMgr);

    m_vp=m_graphicsDevice->getBufferFactoryRef()->createMat4CBuffer();
}

App::~App()
{	
    SAFE_DELETE(m_kernelDevice);
    SAFE_DELETE(m_graphicsDevice);
    SAFE_DELETE(m_context);
    SAFE_DELETE(m_input);
    SAFE_DELETE(m_controller);
    SAFE_DELETE(m_modelImporter);
    SAFE_DELETE(m_sceneMgr);
    //
    delete m_kdDebugBoxInstances;
    delete m_vp;
}

void App::run()
{
    // Set up windows timer
    __int64 countsPerSec = 0;
    __int64 currTimeStamp = 0;
    QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
    double secsPerCount = 1.0f / (float)countsPerSec;

    double dt = 0.0;
    double fps = 0.0f;
    __int64 m_prevTimeStamp = 0;

    QueryPerformanceCounter((LARGE_INTEGER*)&m_prevTimeStamp);
    QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

    MSG msg = {0};

    // secondary run variable
    // lets non-context systems quit the program
    bool run=true;

    int shadowMode=0;
    int debugDrawMode=0;
    bool drawKdTree=false;
    float thrustPowInc=0.0f;

    // load assets
    
    int duck = m_modelImporter->loadFile("../Assets/Swampler_reexport.dae");
    ModelImporter::ModelData* duckMdl=m_modelImporter->getStoredModel(duck);

    aiMesh* mmesh=duckMdl->m_model->mMeshes[0];
    char32_t* indices=&(duckMdl->m_trisIndices[0]);
    m_sceneMgr->addMeshTris(reinterpret_cast<void*>(mmesh->mVertices),
                            mmesh->mNumVertices,
                            indices,
                            duckMdl->m_trisIndices.size(),
                            reinterpret_cast<void*>(mmesh->mNormals),
                            reinterpret_cast<void*>(mmesh->mTextureCoords[0]));
    RawTexture* t=m_modelImporter->getModelTexture(duck);
    m_sceneMgr->addTexture(t);
    
    int treeId=duckMdl->m_treeId;
    vector<KDNode>* KDnodes=m_modelImporter->getKDTree(treeId);
    vector<KDLeaf>* KDleaves=m_modelImporter->getKDLeafList(treeId);
    vector<int>* KDindices=m_modelImporter->getKDLeafDataList(treeId);
    KDBounds KDRootBounds=m_modelImporter->getTreeBounds(treeId);
    m_sceneMgr->addKDTree(KDRootBounds,
                          &(*KDnodes)[0]  ,KDnodes->size(),
                          &(*KDleaves)[0] ,KDleaves->size(),
                          &(*KDindices)[0],KDindices->size());						

    // debug stuff for kd tree
    vector<KDBounds>* kdBounds=m_modelImporter->getDebugNodeBounds(treeId);
    int numBounds=kdBounds->size();
    for (int n=0;n<numBounds;n++)
    {
        glm::mat4 translation=glm::translate(glm::mat4(1.0f),(*kdBounds)[n].m_pos);
        glm::mat4 scale=glm::scale(glm::mat4(1.0f),(*kdBounds)[n].m_extents*0.5f);
        m_kdDebugBoxMats.push_back(glm::transpose(translation*scale));
    }
    m_kdDebugBoxInstances=m_graphicsDevice->getBufferFactoryRef()->createMat4InstanceBuffer((void*)&m_kdDebugBoxMats[0],numBounds);
    
    //

    while (!m_context->closeRequested() && run)
    {
        while( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE)>0 )
        {
			TranslateMessage( &msg );
			DispatchMessage( &msg );
        }
       // else
        {
            m_input->run();
            // get joystate
            //Just dump the current joy state
            JoyStick* joy = nullptr;
            if (m_input->hasJoysticks()) 
                joy = m_input->g_joys[0];
            // Power
            float thrustPow=0.05f;
            if (m_input->g_kb->isKeyDown(KC_LCONTROL)  || (joy!=nullptr && joy->getJoyStickState().mButtons[0]))
            {
                thrustPowInc+=(1.0f+0.001f*thrustPowInc)*(float)dt;
                thrustPow=2.2f+thrustPowInc;
            }
            else
            {
                thrustPowInc=0.0f;
            }
            // Thrust
            if (m_input->g_kb->isKeyDown(KC_LEFT) || m_input->g_kb->isKeyDown(KC_A))
                m_controller->moveThrust(glm::vec3(-1.0f,0.0f,0.0f)*thrustPow);
            if (m_input->g_kb->isKeyDown(KC_RIGHT) || m_input->g_kb->isKeyDown(KC_D))
                m_controller->moveThrust(glm::vec3(1.0f,0.0f,0.0f)*thrustPow);
            if (m_input->g_kb->isKeyDown(KC_UP) || m_input->g_kb->isKeyDown(KC_W))
                m_controller->moveThrust(glm::vec3(0.0f,1.0f,0.0f)*thrustPow);
            if (m_input->g_kb->isKeyDown(KC_DOWN) || m_input->g_kb->isKeyDown(KC_S))
                m_controller->moveThrust(glm::vec3(0.0f,-1.0f,0.0f)*thrustPow);
            if (m_input->g_kb->isKeyDown(KC_SPACE))
                m_controller->moveThrust(glm::vec3(0.0f,0.0f,1.0f)*thrustPow);
            if (m_input->g_kb->isKeyDown(KC_B))
                m_controller->moveThrust(glm::vec3(0.0f,0.0f,-1.0f)*thrustPow);
            // Joy thrust
            if (joy!=nullptr)
            {
                const JoyStickState& js = joy->getJoyStickState();
                m_controller->moveThrust(glm::vec3((float)(invclampcap(js.mAxes[1].abs,-5000,5000))* 0.0001f,
                                                   (float)(invclampcap(js.mAxes[0].abs,-5000,5000))*-0.0001f,
                                                   (float)(js.mAxes[4].abs)*-0.0001f)*thrustPow);
            }


            // Angular thrust
            if (m_input->g_kb->isKeyDown(KC_Q) || (joy!=nullptr && joy->getJoyStickState().mButtons[4]))
                m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,-1.0f));
            if (m_input->g_kb->isKeyDown(KC_E) || (joy!=nullptr && joy->getJoyStickState().mButtons[5]))
                m_controller->moveAngularThrust(glm::vec3(0.0f,0.0f,1.0f));
            /*if (m_input->g_kb->isKeyDown(KC_T))
                m_controller->moveAngularThrust(glm::vec3(0.0f,1.0f,0.0f));
            if (m_input->g_kb->isKeyDown(KC_R))
                m_controller->moveAngularThrust(glm::vec3(0.0f,-1.0f,0.0f));
            if (m_input->g_kb->isKeyDown(KC_U))
                m_controller->moveAngularThrust(glm::vec3(1.0f,0.0f,0.0f));
            if (m_input->g_kb->isKeyDown(KC_J))
                m_controller->moveAngularThrust(glm::vec3(-1.0f,0.0f,0.0f));*/
            // Joy angular thrust
            if (joy!=nullptr)
            {
                const JoyStickState& js = joy->getJoyStickState();
                m_controller->moveAngularThrust(glm::vec3((float)(invclampcap(js.mAxes[2].abs,-5000,5000))*-0.00001f,
                                                          (float)(invclampcap(js.mAxes[3].abs,-5000,5000))*-0.00001f,
                                                          0.0f));
            }
            // Settings
            if (m_input->g_kb->isKeyDown(KC_K)) // Debug blocks
                debugDrawMode=1;
            if (m_input->g_kb->isKeyDown(KC_L)) // Debug off
                debugDrawMode=0;
            if (m_input->g_kb->isKeyDown(KC_J)) // Debug kdtree
                debugDrawMode=2;
            if (m_input->g_kb->isKeyDown(KC_O)) // Debug wireframe
                drawKdTree=true;
            if (m_input->g_kb->isKeyDown(KC_P)) // Debug wireframe off
                drawKdTree=false;
			if (m_input->g_kb->isKeyDown(KC_ADD)) // Resize window auto on
				m_context->setToUpdateOnResize(true);
			if (m_input->g_kb->isKeyDown(KC_SUBTRACT)) // Resize window auto off
				m_context->setToUpdateOnResize(false);
            if (m_input->g_kb->isKeyDown(KC_0)) // Shadow off
                shadowMode=0;
            if (m_input->g_kb->isKeyDown(KC_1)) // Shadow on (hard shadows)
                shadowMode=1;
            if (m_input->g_kb->isKeyDown(KC_2)) // Shadow on (soft shadows fidelity=2)
                shadowMode=2;
            if (m_input->g_kb->isKeyDown(KC_3)) // Shadow on (soft shadows fidelity=5)
                shadowMode=5;
            if (m_input->g_kb->isKeyDown(KC_4)) // Shadow on (soft shadows fidelity=10)
                shadowMode=10;
            if (m_input->g_kb->isKeyDown(KC_5)) // Shadow on (soft shadows fidelity=15)
                shadowMode=15;
            if (m_input->g_kb->isKeyDown(KC_6)) // Shadow on (soft shadows fidelity=20)
                shadowMode=20;

            float mousemovemultiplier=0.001f;
            float mouseX=(float)m_input->g_m->getMouseState().X.rel*mousemovemultiplier;
            float mouseY=(float)m_input->g_m->getMouseState().Y.rel*mousemovemultiplier;
            if (abs(mouseX)>0.0f || abs(mouseY)>0.0f)
            {
                m_controller->rotate(glm::vec3(clamp(-mouseY,-1.0f,1.0f),clamp(-mouseX,-1.0f,1.0f),0.0f));
            }

            // apply resizing on graphics device if it has been triggered by the context
            if (m_context->isSizeDirty())
            {
                pair<int,int> sz=m_context->getSize();
                m_graphicsDevice->updateResolution(sz.first,sz.second);
            }

            // Get Delta time
            QueryPerformanceCounter((LARGE_INTEGER*)&currTimeStamp);

            dt = (currTimeStamp - m_prevTimeStamp) * secsPerCount;
            fps = 1.0f/dt;
            
            dt = clamp(dt,0.0,DTCAP);
            m_prevTimeStamp = currTimeStamp;

            fpsUpdateTick-=(float)dt;
            if (fpsUpdateTick<=0.0f)
            {
                m_context->updateTitle((" | FPS: "+toString((int)fps)).c_str());
                //DEBUGPRINT((("\n"+toString(dt)).c_str())); 
                fpsUpdateTick=0.3f;
            }

            m_graphicsDevice->clearRenderTargets();									// Clear render targets

            // temp controller update code
            m_controller->setFovFromAngle(60.0f+min(thrustPow*0.01f,35.0f),
                                          m_graphicsDevice->getAspectRatio());
            m_controller->update((float)dt);
            std::memcpy(&m_vp->accessBuffer,&m_controller->getViewProjMatrix(),sizeof(float)*4*4);
            m_vp->update();

            // Run the devices
            // ---------------------------------------------------------------------------------------------
            m_kernelDevice->update((float)dt,m_controller,debugDrawMode,shadowMode);	// Update kernel data



            m_kernelDevice->executeKernelJob((float)dt,KernelDevice::J_RAYTRACEWORLD);		// Run kernels

            m_graphicsDevice->executeRenderPass(GraphicsDevice::P_COMPOSEPASS);		// Run passes

            if (drawKdTree)
                m_graphicsDevice->executeRenderPass(GraphicsDevice::P_WIREFRAMEPASS,m_vp,m_kdDebugBoxInstances);		// Run passes
            m_graphicsDevice->flipBackBuffer();										// Flip!
            // ---------------------------------------------------------------------------------------------
        }
    }
}
