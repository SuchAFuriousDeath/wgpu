//! Native desktop OpenGL on macOS through CGL.
//!
//! The adapter owns a headless CGL context (OpenGL 4.1 or 3.2+ core profile) that all GL
//! work happens on. Every [`Surface`] owns a second CGL context in the same share group,
//! attached to the window's `NSView` through an `NSOpenGLContext`. Swapchain images are
//! renderbuffers in the share group; presenting blits the renderbuffer into the surface
//! context's default framebuffer and flushes it, like the WGL backend does with the window DC.
//!
//! Everything is loaded at runtime from `OpenGL.framework`, so an application built with
//! this backend still starts on systems where OpenGL is unavailable.

use alloc::{boxed::Box, ffi::CString, format, string::String, sync::Arc, vec::Vec};
use core::{
    ffi::{c_int, c_void, CStr},
    fmt,
    mem::ManuallyDrop,
    panic::AssertUnwindSafe,
    ptr,
    sync::atomic::{AtomicBool, AtomicI32, Ordering},
    time::Duration,
};

use glow::HasContext;
use objc2::{
    encode::{Encode, Encoding, RefEncode},
    msg_send,
    rc::{autoreleasepool, Allocated, Retained},
    runtime::{AnyClass, AnyObject},
    MainThreadMarker,
};
use parking_lot::{Mutex, MutexGuard, RwLock};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

/// The amount of time to wait while trying to obtain a lock to the adapter context
const CONTEXT_LOCK_TIMEOUT_SECS: u64 = 1;

const OPENGL_FRAMEWORK_PATH: &str = "/System/Library/Frameworks/OpenGL.framework/OpenGL";

mod ffi {
    #![allow(non_upper_case_globals)]

    use core::ffi::{c_char, c_int, c_uint, c_void};

    pub type CGLError = c_int;
    pub type CGLPixelFormatAttribute = c_int;
    pub type CGLContextParameter = c_int;
    pub type CGLPixelFormatObj = *mut c_void;
    pub type CGLContextObj = *mut super::CGLContextObject;

    pub const kCGLNoError: CGLError = 0;

    pub const kCGLPFADoubleBuffer: CGLPixelFormatAttribute = 5;
    pub const kCGLPFAColorSize: CGLPixelFormatAttribute = 8;
    pub const kCGLPFAAlphaSize: CGLPixelFormatAttribute = 11;
    pub const kCGLPFANoRecovery: CGLPixelFormatAttribute = 72;
    pub const kCGLPFAAccelerated: CGLPixelFormatAttribute = 73;
    pub const kCGLPFAAllowOfflineRenderers: CGLPixelFormatAttribute = 96;
    pub const kCGLPFAOpenGLProfile: CGLPixelFormatAttribute = 99;

    /// macOS 10.7+.
    pub const kCGLOGLPVersion_3_2_Core: CGLPixelFormatAttribute = 0x3200;
    /// macOS 10.9+. Older systems reject it with `kCGLBadAttribute`.
    pub const kCGLOGLPVersion_GL4_Core: CGLPixelFormatAttribute = 0x4100;

    pub const kCGLCPSwapInterval: CGLContextParameter = 222;

    pub type ChoosePixelFormat = unsafe extern "C" fn(
        attribs: *const CGLPixelFormatAttribute,
        pix: *mut CGLPixelFormatObj,
        npix: *mut c_int,
    ) -> CGLError;
    pub type ReleasePixelFormat = unsafe extern "C" fn(pix: CGLPixelFormatObj);
    pub type CreateContext = unsafe extern "C" fn(
        pix: CGLPixelFormatObj,
        share: CGLContextObj,
        ctx: *mut CGLContextObj,
    ) -> CGLError;
    pub type ReleaseContext = unsafe extern "C" fn(ctx: CGLContextObj);
    pub type RetainContext = unsafe extern "C" fn(ctx: CGLContextObj) -> CGLContextObj;
    pub type GetContextRetainCount = unsafe extern "C" fn(ctx: CGLContextObj) -> c_uint;
    pub type SetCurrentContext = unsafe extern "C" fn(ctx: CGLContextObj) -> CGLError;
    pub type GetCurrentContext = unsafe extern "C" fn() -> CGLContextObj;
    pub type SetParameter = unsafe extern "C" fn(
        ctx: CGLContextObj,
        pname: CGLContextParameter,
        params: *const c_int,
    ) -> CGLError;
    pub type SetVirtualScreen = unsafe extern "C" fn(ctx: CGLContextObj, screen: c_int) -> CGLError;
    pub type GetVirtualScreen =
        unsafe extern "C" fn(ctx: CGLContextObj, screen: *mut c_int) -> CGLError;
    pub type LockContext = unsafe extern "C" fn(ctx: CGLContextObj) -> CGLError;
    pub type ErrorString = unsafe extern "C" fn(error: CGLError) -> *const c_char;

    unsafe extern "C" {
        // Both are part of libSystem, which every macOS binary links.
        pub static _dispatch_main_q: c_void;
        pub fn dispatch_async_f(
            queue: *const c_void,
            context: *mut c_void,
            work: unsafe extern "C" fn(*mut c_void),
        );
    }
}

/// Opaque `struct _CGLContextObject`.
///
/// Declared with its real Objective-C encoding so that `objc2` accepts it as the argument of
/// `-[NSOpenGLContext initWithCGLContextObj:]` when it verifies message encodings.
#[repr(C)]
pub struct CGLContextObject {
    _priv: [u8; 0],
}

unsafe impl RefEncode for CGLContextObject {
    const ENCODING_REF: Encoding = Encoding::Pointer(&Encoding::Struct("_CGLContextObject", &[]));
}

/// The CGL entry points used by this backend, together with the library they came from.
struct Cgl {
    library: libloading::Library,
    choose_pixel_format: ffi::ChoosePixelFormat,
    release_pixel_format: ffi::ReleasePixelFormat,
    create_context: ffi::CreateContext,
    release_context: ffi::ReleaseContext,
    retain_context: ffi::RetainContext,
    get_context_retain_count: ffi::GetContextRetainCount,
    set_current_context: ffi::SetCurrentContext,
    get_current_context: ffi::GetCurrentContext,
    set_parameter: ffi::SetParameter,
    set_virtual_screen: ffi::SetVirtualScreen,
    get_virtual_screen: ffi::GetVirtualScreen,
    lock_context: ffi::LockContext,
    unlock_context: ffi::LockContext,
    error_string: ffi::ErrorString,
}

impl fmt::Debug for Cgl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cgl").finish_non_exhaustive()
    }
}

impl Cgl {
    fn load() -> Result<Self, crate::InstanceError> {
        let library = unsafe { libloading::Library::new(OPENGL_FRAMEWORK_PATH) }.map_err(|e| {
            crate::InstanceError::with_source(String::from("unable to open OpenGL.framework"), e)
        })?;

        fn sym<T: Copy>(
            library: &libloading::Library,
            name: &CStr,
        ) -> Result<T, crate::InstanceError> {
            unsafe { library.get::<T>(name.to_bytes_with_nul()) }
                .map(|symbol| *symbol)
                .map_err(|e| {
                    crate::InstanceError::with_source(
                        format!("OpenGL.framework is missing {name:?}"),
                        e,
                    )
                })
        }

        Ok(Self {
            choose_pixel_format: sym(&library, c"CGLChoosePixelFormat")?,
            release_pixel_format: sym(&library, c"CGLReleasePixelFormat")?,
            create_context: sym(&library, c"CGLCreateContext")?,
            release_context: sym(&library, c"CGLReleaseContext")?,
            retain_context: sym(&library, c"CGLRetainContext")?,
            get_context_retain_count: sym(&library, c"CGLGetContextRetainCount")?,
            set_current_context: sym(&library, c"CGLSetCurrentContext")?,
            get_current_context: sym(&library, c"CGLGetCurrentContext")?,
            set_parameter: sym(&library, c"CGLSetParameter")?,
            set_virtual_screen: sym(&library, c"CGLSetVirtualScreen")?,
            get_virtual_screen: sym(&library, c"CGLGetVirtualScreen")?,
            lock_context: sym(&library, c"CGLLockContext")?,
            unlock_context: sym(&library, c"CGLUnlockContext")?,
            error_string: sym(&library, c"CGLErrorString")?,
            library,
        })
    }

    fn error_message(&self, error: ffi::CGLError) -> String {
        let message = unsafe { (self.error_string)(error) };
        if message.is_null() {
            format!("CGL error {error}")
        } else {
            let message = unsafe { CStr::from_ptr(message) }.to_string_lossy();
            format!("{message} ({error})")
        }
    }

    fn check(&self, error: ffi::CGLError) -> Result<(), String> {
        if error == ffi::kCGLNoError {
            Ok(())
        } else {
            Err(self.error_message(error))
        }
    }

    fn get_proc_address(&self, name: &str) -> *const c_void {
        let Ok(name) = CString::new(name) else {
            return ptr::null();
        };
        unsafe { self.library.get::<*const c_void>(name.as_bytes_with_nul()) }
            .map_or(ptr::null(), |symbol| *symbol)
    }
}

struct PixelFormat {
    cgl: Arc<Cgl>,
    raw: ffi::CGLPixelFormatObj,
}

impl PixelFormat {
    fn choose(cgl: &Arc<Cgl>) -> Result<Self, crate::InstanceError> {
        let profiles = [ffi::kCGLOGLPVersion_GL4_Core, ffi::kCGLOGLPVersion_3_2_Core];
        let mut last_error = String::from("no pixel format matched");

        // Prefer a hardware renderer, but fall back to Apple's software renderer rather
        // than failing outright. The software renderer is then chosen explicitly, and shows up
        // as a CPU adapter.
        for accelerated in [true, false] {
            for profile in profiles {
                let mut attributes = Vec::with_capacity(12);
                attributes.extend([ffi::kCGLPFAOpenGLProfile, profile]);
                attributes.extend([ffi::kCGLPFAColorSize, 24, ffi::kCGLPFAAlphaSize, 8]);
                attributes.push(ffi::kCGLPFADoubleBuffer);
                // Lets the context follow the window to another GPU on multi-GPU Macs and
                // keeps the integrated GPU usable when the app opts into automatic graphics
                // switching in its `Info.plist`.
                attributes.push(ffi::kCGLPFAAllowOfflineRenderers);
                if accelerated {
                    // Without `NoRecovery`, CGL may silently move a hardware context to the
                    // software renderer, e.g. when the GPU runs out of memory, and keep
                    // reporting the hardware adapter's capabilities meanwhile.
                    attributes.extend([ffi::kCGLPFAAccelerated, ffi::kCGLPFANoRecovery]);
                }
                attributes.push(0);

                let mut raw = ptr::null_mut();
                let mut count: c_int = 0;
                let result =
                    unsafe { (cgl.choose_pixel_format)(attributes.as_ptr(), &mut raw, &mut count) };
                match cgl.check(result) {
                    Ok(()) if !raw.is_null() => {
                        log::debug!(
                            "Chose CGL pixel format: profile {profile:#x}, accelerated: {accelerated}, \
                             {count} virtual screen(s)"
                        );
                        return Ok(Self {
                            cgl: Arc::clone(cgl),
                            raw,
                        });
                    }
                    Ok(()) => {}
                    Err(e) => last_error = e,
                }
            }
        }

        Err(crate::InstanceError::new(format!(
            "unable to choose a CGL pixel format: {last_error}"
        )))
    }
}

impl Drop for PixelFormat {
    fn drop(&mut self) {
        unsafe { (self.cgl.release_pixel_format)(self.raw) };
    }
}

unsafe impl Send for PixelFormat {}
unsafe impl Sync for PixelFormat {}

/// An owned reference to a CGL context.
struct CglContext {
    cgl: Arc<Cgl>,
    raw: ffi::CGLContextObj,
}

impl CglContext {
    fn new(pixel_format: &PixelFormat, share: Option<&CglContext>) -> Result<Self, String> {
        let cgl = &pixel_format.cgl;
        let mut raw = ptr::null_mut();
        let share = share.map_or(ptr::null_mut(), |share| share.raw);
        cgl.check(unsafe { (cgl.create_context)(pixel_format.raw, share, &mut raw) })?;
        if raw.is_null() {
            return Err(String::from("CGLCreateContext returned no context"));
        }
        Ok(Self {
            cgl: Arc::clone(cgl),
            raw,
        })
    }

    fn make_current(&self) -> Result<(), String> {
        self.cgl
            .check(unsafe { (self.cgl.set_current_context)(self.raw) })
    }

    fn unmake_current(&self) -> Result<(), String> {
        if unsafe { (self.cgl.get_current_context)() }.is_null() {
            return Ok(());
        }
        self.cgl
            .check(unsafe { (self.cgl.set_current_context)(ptr::null_mut()) })
    }

    fn virtual_screen(&self) -> Result<c_int, String> {
        let mut screen = 0;
        self.cgl
            .check(unsafe { (self.cgl.get_virtual_screen)(self.raw, &mut screen) })?;
        Ok(screen)
    }

    /// Takes the CGL lock of this context. It serializes access between threads that use the
    /// same context, e.g. a render thread presenting while the main thread calls `update`.
    fn lock(&self) -> CglContextLock<'_> {
        if let Err(e) = self.cgl.check(unsafe { (self.cgl.lock_context)(self.raw) }) {
            log::error!("CGLLockContext failed: {e}");
        }
        CglContextLock { context: self }
    }
}

impl Drop for CglContext {
    fn drop(&mut self) {
        unsafe { (self.cgl.release_context)(self.raw) };
    }
}

unsafe impl Send for CglContext {}
unsafe impl Sync for CglContext {}

struct CglContextLock<'a> {
    context: &'a CglContext,
}

impl Drop for CglContextLock<'_> {
    fn drop(&mut self) {
        let cgl = &self.context.cgl;
        if let Err(e) = cgl.check(unsafe { (cgl.unlock_context)(self.context.raw) }) {
            log::error!("CGLUnlockContext failed: {e}");
        }
    }
}

/// A wrapper around a [`glow::Context`] and the required CGL context that uses locking to
/// guarantee exclusive access when shared with multiple threads.
pub struct AdapterContext {
    inner: Arc<Mutex<Inner>>,
    /// The `CGLContextObj` owned by `inner`, or null for an external adapter.
    ///
    /// Kept outside the mutex so that it can be read while the context is locked, e.g. by
    /// [`super::Adapter::expose`].
    raw: ffi::CGLContextObj,
}

impl fmt::Debug for AdapterContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterContext").finish_non_exhaustive()
    }
}

// SAFETY: `raw` is only an identifier here; the context it points to is owned by `inner` and
// only used through the mutex.
unsafe impl Send for AdapterContext {}
unsafe impl Sync for AdapterContext {}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(AdapterContext: Send, Sync);

impl AdapterContext {
    /// `raw` must be the context owned by `inner`, or null if it owns none. It's passed in
    /// rather than read from `inner`, so that this never waits for the lock.
    fn new(inner: Arc<Mutex<Inner>>, raw: ffi::CGLContextObj) -> Self {
        Self { inner, raw }
    }

    /// Returns whether wgpu-hal created and owns the CGL context, i.e. whether this is not an
    /// external adapter.
    ///
    /// Doesn't lock the context, so it can be called while holding [`Self::lock`].
    pub fn is_owned(&self) -> bool {
        !self.raw.is_null()
    }

    /// Returns the underlying `CGLContextObj`, or null for an external adapter.
    ///
    /// Doesn't lock the context, so it can be called while holding [`Self::lock`].
    pub fn raw_context(&self) -> *mut c_void {
        self.raw.cast()
    }

    /// Obtain a lock to the CGL context and get handle to the [`glow::Context`] that can be
    /// used to do rendering.
    #[track_caller]
    pub fn lock(&self) -> AdapterContextLock<'_> {
        // Don't lock forever. If it takes longer than 1 second to get the lock we've got a
        // deadlock and should panic to show where we got stuck
        let inner = try_lock_inner(&self.inner)
            .expect("Could not lock adapter context. This is most-likely a deadlock.");

        if let Some(context) = &inner.context {
            context.make_current().unwrap();
        }

        AdapterContextLock { inner }
    }
}

/// A guard containing a lock to an [`AdapterContext`], while the GL context is kept current.
pub struct AdapterContextLock<'a> {
    inner: MutexGuard<'a, Inner>,
}

impl fmt::Debug for AdapterContextLock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterContextLock").finish_non_exhaustive()
    }
}

fn try_lock_inner(inner: &Mutex<Inner>) -> Option<MutexGuard<'_, Inner>> {
    inner.try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
}

impl AdapterContextLock<'_> {
    /// Makes `surface_context` current, runs `f`, and makes the previously current context
    /// (normally the adapter's) current again.
    ///
    /// On macOS, GL entry points dispatch through the thread's current CGL context, so the
    /// adapter's [`glow::Context`] drives whichever context is current.
    fn with_surface_context<R>(
        &self,
        surface_context: &CglContext,
        f: impl FnOnce(&glow::Context) -> R,
    ) -> Result<R, String> {
        // Commands issued by one context only become visible to another context of the share
        // group after they are flushed.
        unsafe { self.inner.gl.flush() };

        let cgl = &surface_context.cgl;
        let previous = unsafe { (cgl.get_current_context)() };
        let _cgl_lock = surface_context.lock();
        surface_context.make_current()?;
        let result = f(&self.inner.gl);
        cgl.check(unsafe { (cgl.set_current_context)(previous) })?;
        Ok(result)
    }

    /// Moves the adapter context to `virtual_screen`, i.e. to the renderer (GPU) that a surface
    /// context presents with after its window moved to another display.
    ///
    /// Apple asks to keep contexts that share objects with an on-screen context on the same
    /// virtual screen as it, so that the driver doesn't need to shuttle every shared object
    /// between GPUs. Chromium does the same for its offscreen contexts. Unverified on real
    /// multi-GPU hardware: the adapter's limits and features stay those of the renderer it was
    /// created on, which may differ from the new renderer's.
    fn follow_virtual_screen(&self, virtual_screen: c_int) {
        let Some(context) = &self.inner.context else {
            return;
        };
        match context.virtual_screen() {
            Ok(current) if current == virtual_screen => {}
            Ok(current) => {
                log::info!(
                    "Moving the CGL adapter context from virtual screen {current} to \
                     {virtual_screen}"
                );
                unsafe { self.inner.gl.flush() };
                let cgl = &context.cgl;
                if let Err(e) =
                    cgl.check(unsafe { (cgl.set_virtual_screen)(context.raw, virtual_screen) })
                {
                    log::warn!("unable to set the virtual screen of the adapter context: {e}");
                }
            }
            Err(e) => log::warn!("unable to get the virtual screen of the adapter context: {e}"),
        }
    }
}

impl core::ops::Deref for AdapterContextLock<'_> {
    type Target = glow::Context;

    fn deref(&self) -> &Self::Target {
        &self.inner.gl
    }
}

impl Drop for AdapterContextLock<'_> {
    fn drop(&mut self) {
        if let Some(context) = &self.inner.context {
            context.unmake_current().unwrap()
        }
    }
}

struct Inner {
    gl: ManuallyDrop<glow::Context>,
    /// `None` for adapters created through [`super::Adapter::new_external`].
    context: Option<CglContext>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        struct CurrentGuard<'a>(&'a CglContext);
        impl Drop for CurrentGuard<'_> {
            fn drop(&mut self) {
                self.0.unmake_current().unwrap();
            }
        }

        // Context must be current when dropped. See safety docs on
        // `glow::HasContext`.
        //
        // NOTE: This is only set to `None` by `Adapter::new_external` which
        // requires the context to be current when anything that may be holding
        // the `Arc<AdapterShared>` is dropped.
        let _guard = self.context.as_ref().map(|context| {
            context.make_current().unwrap();
            CurrentGuard(context)
        });
        // SAFETY: Field not used after this.
        unsafe { ManuallyDrop::drop(&mut self.gl) };
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

pub struct Instance {
    pixel_format: Arc<PixelFormat>,
    options: wgt::GlBackendOptions,
    inner: Arc<Mutex<Inner>>,
    /// The context owned by `inner`.
    raw_context: ffi::CGLContextObj,
}

// SAFETY: See `AdapterContext`.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instance").finish_non_exhaustive()
    }
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(Instance: Send, Sync);

impl crate::Instance for Instance {
    type A = super::Api;

    unsafe fn init(desc: &crate::InstanceDescriptor<'_>) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init OpenGL (CGL) Backend");
        let cgl = Arc::new(Cgl::load()?);
        let pixel_format = Arc::new(PixelFormat::choose(&cgl)?);

        let context = CglContext::new(&pixel_format, None)
            .map_err(|e| crate::InstanceError::new(format!("unable to create CGL context: {e}")))?;
        context.make_current().map_err(|e| {
            crate::InstanceError::new(format!("unable to make the CGL context current: {e}"))
        })?;

        let mut gl =
            unsafe { glow::Context::from_loader_function(|name| cgl.get_proc_address(name)) };

        // In contrast to OpenGL ES, OpenGL requires explicitly enabling sRGB conversions,
        // as otherwise the user has to do the sRGB conversion. `GL_FRAMEBUFFER_SRGB` is core
        // since OpenGL 3.0, so every core profile context supports it.
        unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };

        // macOS doesn't implement `KHR_debug`, but keep this in case a future driver does.
        if desc.flags.contains(wgt::InstanceFlags::VALIDATION) && gl.supports_debug() {
            log::debug!("Enabling GL debug output");
            unsafe { gl.enable(glow::DEBUG_OUTPUT) };
            unsafe { gl.debug_message_callback(super::gl_debug_message_callback) };
        }

        // Wrap in ManuallyDrop to make it easier to "current" the GL context before dropping this
        // GLOW context, which could also happen if a panic occurs after we uncurrent the context
        // below but before Inner is constructed.
        let gl = ManuallyDrop::new(gl);
        context.unmake_current().map_err(|e| {
            crate::InstanceError::new(format!("unable to unset the current CGL context: {e}"))
        })?;

        Ok(Instance {
            pixel_format,
            options: desc.backend_options.gl.clone(),
            raw_context: context.raw,
            inner: Arc::new(Mutex::new(Inner {
                gl,
                context: Some(context),
            })),
        })
    }

    unsafe fn create_surface(
        &self,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Surface, crate::InstanceError> {
        if !matches!(display_handle, RawDisplayHandle::AppKit(_)) {
            return Err(crate::InstanceError::new(format!(
                "unsupported display: {display_handle:?}"
            )));
        }
        let RawWindowHandle::AppKit(handle) = window_handle else {
            return Err(crate::InstanceError::new(format!(
                "unsupported window: {window_handle:?}"
            )));
        };

        // The share context is only safe to use while no other thread renders with it. Don't
        // block forever if the calling thread already holds the adapter context lock.
        let inner = try_lock_inner(&self.inner).ok_or_else(|| {
            crate::InstanceError::new(String::from(
                "timed out waiting for the adapter context lock",
            ))
        })?;
        let share = inner.context.as_ref();
        let context = CglContext::new(&self.pixel_format, share).map_err(|e| {
            crate::InstanceError::new(format!("unable to create surface CGL context: {e}"))
        })?;
        drop(inner);

        // SAFETY: The caller guarantees that the window handle is valid.
        let view = unsafe { Retained::retain(handle.ns_view.as_ptr().cast::<AnyObject>()) }
            .ok_or_else(|| crate::InstanceError::new(String::from("the NSView is null")))?;
        let ns_context = new_ns_opengl_context(&context)?;
        let state = Arc::new(ViewState {
            ns_context,
            view,
            context,
            attached: AtomicBool::new(false),
            refresh_queued: AtomicBool::new(false),
            drawable_size: Mutex::new(None),
            virtual_screen: AtomicI32::new(-1),
            screen: Mutex::new(None),
        });

        state.queue_refresh();

        Ok(Surface {
            state: ManuallyDrop::new(state),
            presentable: true,
            swapchain: RwLock::new(None),
        })
    }

    unsafe fn enumerate_adapters(
        &self,
        _surface_hint: Option<&Surface>,
    ) -> Vec<crate::ExposedAdapter<super::Api>> {
        unsafe {
            super::Adapter::expose(
                AdapterContext::new(Arc::clone(&self.inner), self.raw_context),
                self.options.clone(),
            )
        }
        .into_iter()
        .collect()
    }
}

impl super::Adapter {
    /// Creates a new external adapter using the specified loader function.
    ///
    /// # Safety
    ///
    /// - The underlying OpenGL context must be current.
    /// - The underlying OpenGL context must be current when interfacing with any objects returned by
    ///   wgpu-hal from this adapter.
    /// - The underlying OpenGL context must be current when dropping this adapter and when
    ///   dropping any objects returned from this adapter.
    pub unsafe fn new_external(
        fun: impl FnMut(&str) -> *const c_void,
        options: wgt::GlBackendOptions,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        let context = unsafe { glow::Context::from_loader_function(fun) };
        unsafe {
            Self::expose(
                AdapterContext::new(
                    Arc::new(Mutex::new(Inner {
                        gl: ManuallyDrop::new(context),
                        context: None,
                    })),
                    ptr::null_mut(),
                ),
                options,
            )
        }
    }

    pub fn adapter_context(&self) -> &AdapterContext {
        &self.shared.context
    }
}

impl super::Device {
    /// Returns the underlying CGL context.
    pub fn context(&self) -> &AdapterContext {
        &self.shared.context
    }
}

/// Runs `f` on the main thread: right away if this is the main thread, otherwise later, on the
/// main dispatch queue, without waiting for it.
///
/// AppKit requires `NSOpenGLContext`'s view management (`setView:`, `update`, `clearDrawable`)
/// to happen on the main thread. Waiting for the main thread instead would deadlock whenever
/// it waits for the calling thread, e.g. while it joins a render thread that drops a surface,
/// or after its run loop stopped. The flip side is that work queued from another thread only
/// runs once the main thread's run loop gets to it, and never if it doesn't run anymore.
fn run_on_main<F: FnOnce(MainThreadMarker) + Send + 'static>(f: F) {
    if let Some(mtm) = MainThreadMarker::new() {
        f(mtm);
        return;
    }

    unsafe extern "C" fn trampoline<F: FnOnce(MainThreadMarker)>(context: *mut c_void) {
        // SAFETY: `context` is the box leaked below, and the queue runs every block once.
        let f = unsafe { Box::from_raw(context.cast::<F>()) };
        // SAFETY: The main queue only runs on the main thread.
        let mtm = unsafe { MainThreadMarker::new_unchecked() };
        // Unwinding out of this `extern "C"` function would abort the process. There's no
        // caller left to hand a panic to, so log it and leave the main thread running.
        if std::panic::catch_unwind(AssertUnwindSafe(move || (*f)(mtm))).is_err() {
            log::error!("A panic occurred in work that the CGL backend ran on the main thread");
        }
    }

    let context = Box::into_raw(Box::new(f));
    unsafe {
        ffi::dispatch_async_f(
            ptr::addr_of!(ffi::_dispatch_main_q),
            context.cast(),
            trampoline::<F>,
        )
    };
}

/// Creates an `NSOpenGLContext` wrapping `context`, used to attach it to a view.
fn new_ns_opengl_context(
    context: &CglContext,
) -> Result<Retained<AnyObject>, crate::InstanceError> {
    let class = AnyClass::get(c"NSOpenGLContext").ok_or_else(|| {
        crate::InstanceError::new(String::from(
            "NSOpenGLContext is unavailable; is AppKit loaded?",
        ))
    })?;

    let cgl = &context.cgl;
    let retain_count = unsafe { (cgl.get_context_retain_count)(context.raw) };
    let raw: Option<Retained<AnyObject>> = autoreleasepool(|_| unsafe {
        let allocated: Allocated<AnyObject> = msg_send![class, alloc];
        msg_send![allocated, initWithCGLContextObj: context.raw]
    });
    let raw = raw.ok_or_else(|| {
        crate::InstanceError::new(String::from(
            "unable to create an NSOpenGLContext for the surface",
        ))
    })?;

    // `CglContext` keeps its own reference to the CGL context. Apple doesn't document
    // whether `NSOpenGLContext` retains the context it wraps, so if it didn't, give it a
    // reference of its own. Should it then never release that reference, the context
    // leaks, which beats the over-release we'd risk otherwise.
    let new_retain_count = unsafe { (cgl.get_context_retain_count)(context.raw) };
    if new_retain_count <= retain_count {
        log::debug!(
            "NSOpenGLContext didn't retain its CGL context (retain count \
             {retain_count} -> {new_retain_count}), retaining it on its behalf"
        );
        unsafe { (cgl.retain_context)(context.raw) };
    }

    Ok(raw)
}

/// The AppKit side of a [`Surface`], together with the surface's CGL context.
///
/// The `NSOpenGLContext` and the view are messaged on the main thread, through
/// [`run_on_main`], except for reading the view's window and its state (see
/// [`Self::check_window`]) and for `flushBuffer`, which AppKit allows on any thread as long as
/// the CGL context is locked. Other threads otherwise only retain and release this, and the
/// surface hands its own reference to the main thread when it's dropped, so the last
/// reference, and with it the view's, is released there too.
struct ViewState {
    /// Declared before `context`, so that it gives up its reference to the CGL context first.
    ns_context: Retained<AnyObject>,
    /// The `NSView` the surface presents to. Retained, so that work queued on the main thread
    /// can't outlive it.
    view: Retained<AnyObject>,
    context: CglContext,
    /// Whether the context has a drawable on `view`: set by [`Self::refresh`], and cleared
    /// when presenting finds no drawable.
    attached: AtomicBool,
    /// Whether a [`Self::refresh`] is queued on the main thread and didn't start yet.
    refresh_queued: AtomicBool,
    /// The size of the drawable in pixels, as last seen by [`Self::refresh`].
    drawable_size: Mutex<Option<(u32, u32)>>,
    /// The virtual screen of `context`, as last seen by [`Self::refresh`], or -1.
    virtual_screen: AtomicI32,
    /// The `NSScreen` that the view's window was on when [`Self::check_window`] last ran.
    screen: Mutex<Option<Retained<AnyObject>>>,
}

// SAFETY: See the type's documentation.
unsafe impl Send for ViewState {}
unsafe impl Sync for ViewState {}

impl ViewState {
    /// Runs [`Self::refresh`] on the main thread, unless one is queued already.
    ///
    /// Right away on the main thread, otherwise asynchronously: see [`run_on_main`].
    fn queue_refresh(self: &Arc<Self>) {
        if self.refresh_queued.swap(true, Ordering::AcqRel) {
            return;
        }
        let state = Arc::clone(self);
        run_on_main(move |mtm| {
            state.refresh_queued.store(false, Ordering::Release);
            state.refresh(mtm);
        });
    }

    /// Returns whether any part of the view's window is visible, and queues a
    /// [`Self::refresh`] after the window moved to another display, which can mean another
    /// GPU.
    ///
    /// AppKit expects an `update` of the context after such a move, normally in response to
    /// `NSViewGlobalFrameDidChangeNotification`, which nothing observes here, and winit reports
    /// a window that only moved as `Moved`, not `Resized`, so applications don't reconfigure
    /// the surface either.
    ///
    /// Reads the window's state from the calling thread, like the Metal backend does.
    fn check_window(self: &Arc<Self>) -> bool {
        const NS_WINDOW_OCCLUSION_STATE_VISIBLE: usize = 1 << 1;

        let (screen, visible) = autoreleasepool(|_| unsafe {
            let window: Option<Retained<AnyObject>> = msg_send![&*self.view, window];
            match window {
                Some(window) => {
                    let screen: Option<Retained<AnyObject>> = msg_send![&*window, screen];
                    let occlusion_state: usize = msg_send![&*window, occlusionState];
                    (
                        screen,
                        occlusion_state & NS_WINDOW_OCCLUSION_STATE_VISIBLE != 0,
                    )
                }
                None => (None, false),
            }
        });
        let mut last_screen = self.screen.lock();
        let changed =
            last_screen.as_ref().map(Retained::as_ptr) != screen.as_ref().map(Retained::as_ptr);
        if changed {
            *last_screen = screen;
            drop(last_screen);
            self.queue_refresh();
        }
        visible
    }

    /// Presents what was blitted into the drawable.
    ///
    /// Uses `-[NSOpenGLContext flushBuffer]` rather than `CGLFlushDrawable`, since AppKit
    /// presents layer-backed views through its own path. Every view is layer-backed in
    /// applications built against the macOS 10.14 SDK or later.
    fn flush_buffer(&self) -> Result<(), String> {
        let cgl = &self.context.cgl;
        let _cgl_lock = self.context.lock();
        let previous = unsafe { (cgl.get_current_context)() };
        self.context.make_current()?;
        autoreleasepool(|_| unsafe {
            let _: () = msg_send![&*self.ns_context, flushBuffer];
        });
        cgl.check(unsafe { (cgl.set_current_context)(previous) })
    }

    /// Attaches the context to the view, or updates its drawable after the view's size,
    /// position or display changed.
    fn refresh(&self, _mtm: MainThreadMarker) {
        let _cgl_lock = self.context.lock();
        let attached = autoreleasepool(|_| unsafe {
            // AppKit refuses to attach a context to a view whose window has no window device
            // yet, i.e. was never on screen ("invalid drawable"). Don't try before then.
            let window: *mut AnyObject = msg_send![&*self.view, window];
            if window.is_null() {
                return false;
            }
            let window_number: isize = msg_send![window, windowNumber];
            if window_number <= 0 {
                return false;
            }

            let current_view: *mut AnyObject = msg_send![&*self.ns_context, view];
            let view = Retained::as_ptr(&self.view).cast_mut();
            if current_view == view && self.attached.load(Ordering::Acquire) {
                let _: () = msg_send![&*self.ns_context, update];
            } else {
                // Either never attached, or presenting found no drawable although a view is
                // set: attach from scratch.
                if !current_view.is_null() {
                    let _: () = msg_send![&*self.ns_context, clearDrawable];
                }
                // Render at the display's resolution on Retina displays, so that the drawable
                // matches the view's backing size, as wgpu expects from surfaces. winit does
                // this too, but other windowing libraries may not.
                let _: () = msg_send![&*self.view, setWantsBestResolutionOpenGLSurface: true];
                let _: () = msg_send![&*self.ns_context, setView: &*self.view];
            }

            let current_view: *mut AnyObject = msg_send![&*self.ns_context, view];
            if current_view != view {
                return false;
            }

            match self.context.virtual_screen() {
                Ok(screen) => self.virtual_screen.store(screen, Ordering::Release),
                Err(e) => {
                    log::warn!("unable to get the virtual screen of the surface context: {e}")
                }
            }

            let bounds: CGRect = msg_send![&*self.view, bounds];
            let backing: CGRect = msg_send![&*self.view, convertRectToBacking: bounds];
            *self.drawable_size.lock() = Some((
                backing.size.width.round() as u32,
                backing.size.height.round() as u32,
            ));
            true
        });
        if !attached {
            log::debug!("the surface's view isn't on screen yet, will retry attaching to it");
        }
        self.attached.store(attached, Ordering::Release);
    }

    fn detach(&self, _mtm: MainThreadMarker) {
        let _cgl_lock = self.context.lock();
        autoreleasepool(|_| unsafe {
            let _: () = msg_send![&*self.ns_context, clearDrawable];
        });
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

/// `NSRect` on 64-bit macOS.
#[repr(C)]
#[derive(Clone, Copy)]
struct CGRect {
    origin: CGPoint,
    size: CGSize,
}

unsafe impl Encode for CGPoint {
    const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
}

unsafe impl Encode for CGSize {
    const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
}

unsafe impl Encode for CGRect {
    const ENCODING: Encoding = Encoding::Struct("CGRect", &[CGPoint::ENCODING, CGSize::ENCODING]);
}

#[derive(Debug)]
pub struct Swapchain {
    /// Lives in the share group and is rendered to by the adapter context.
    renderbuffer: glow::Renderbuffer,
    /// Framebuffer objects aren't shared, so this one belongs to the surface context.
    framebuffer: glow::Framebuffer,

    extent: wgt::Extent3d,
    format: wgt::TextureFormat,
    format_desc: super::TextureFormatDesc,
}

/// A surface presenting to an `NSView`.
///
/// It can be used from any thread, but it relies on the main thread's run loop: attaching to
/// the view, picking up size changes and detaching on drop happen on the main thread. When
/// called from another thread, these are queued there without waiting for them, and only take
/// effect once the main thread gets to them.
pub struct Surface {
    state: ManuallyDrop<Arc<ViewState>>,
    pub(super) presentable: bool,
    swapchain: RwLock<Option<Swapchain>>,
}

impl fmt::Debug for Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Surface")
            .field("presentable", &self.presentable)
            .field("swapchain", &self.swapchain)
            .finish_non_exhaustive()
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // SAFETY: Field not used after this.
        let state = unsafe { ManuallyDrop::take(&mut self.state) };
        run_on_main(move |mtm| {
            state.detach(mtm);
            drop(state);
        });
    }
}

impl Surface {
    pub(super) unsafe fn present(
        &self,
        _suf_texture: super::Texture,
        context: &AdapterContext,
    ) -> Result<(), crate::SurfaceError> {
        let swapchain = self.swapchain.read();
        let sc = swapchain
            .as_ref()
            .ok_or(crate::SurfaceError::Other("surface isn't configured"))?;
        let gl = context.lock();

        let virtual_screen = self.state.virtual_screen.load(Ordering::Acquire);
        if virtual_screen >= 0 {
            gl.follow_virtual_screen(virtual_screen);
        }

        // The drawable can lag behind the configured size while the view is being resized,
        // and differs from it if the application configured the surface with another size
        // than the view's. Stretch to it then, like `CAMetalLayer` does.
        let (width, height) = (sc.extent.width as i32, sc.extent.height as i32);
        let (dst_width, dst_height) = self
            .state
            .drawable_size
            .lock()
            .map_or((width, height), |(w, h)| (w as i32, h as i32));
        let filter = if (dst_width, dst_height) == (width, height) {
            glow::NEAREST
        } else {
            glow::LINEAR
        };

        let result = gl.with_surface_context(&self.state.context, |gl| {
            unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(sc.framebuffer)) };
            // Reattach every frame: a context only reliably observes changes another context of
            // the share group made to an object after binding it again.
            unsafe {
                gl.framebuffer_renderbuffer(
                    glow::READ_FRAMEBUFFER,
                    glow::COLOR_ATTACHMENT0,
                    glow::RENDERBUFFER,
                    Some(sc.renderbuffer),
                )
            };
            unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };

            // Without a drawable, the default framebuffer is `GL_FRAMEBUFFER_UNDEFINED`.
            let draw_status = unsafe { gl.check_framebuffer_status(glow::DRAW_FRAMEBUFFER) };
            let read_status = unsafe { gl.check_framebuffer_status(glow::READ_FRAMEBUFFER) };
            if draw_status != glow::FRAMEBUFFER_COMPLETE
                || read_status != glow::FRAMEBUFFER_COMPLETE
            {
                unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };
                return Err((draw_status, read_status));
            }

            // Disable sRGB conversions for the blit, like the WGL backend does, so that it copies
            // the bytes of sRGB swapchain formats as they are. The surface context never enables
            // `GL_FRAMEBUFFER_SRGB`, but say so explicitly. Caveat, shared with WGL: before GL
            // 4.4, the spec didn't tie decoding an sRGB read buffer to `GL_FRAMEBUFFER_SRGB`, so
            // a driver may still decode it here and present sRGB formats too dark.
            unsafe { gl.disable(glow::FRAMEBUFFER_SRGB) };

            // Note the Y-flipping here. GL's presentation is not flipped,
            // but main rendering is. Therefore, we Y-flip the output positions
            // in the shader, and also this blit.
            unsafe {
                gl.blit_framebuffer(
                    0,
                    height,
                    width,
                    0,
                    0,
                    0,
                    dst_width,
                    dst_height,
                    glow::COLOR_BUFFER_BIT,
                    filter,
                )
            };
            unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

            Ok(())
        });
        // Flushing may block until the next display refresh with vsync. Don't hold the adapter
        // context lock meanwhile: other threads waiting for it give up after a second.
        drop(gl);

        match result {
            Ok(Ok(())) => self.state.flush_buffer().map_err(|e| {
                log::error!("unable to flush the surface drawable: {e}");
                crate::SurfaceError::Other("unable to flush the surface drawable")
            }),
            Ok(Err((draw_status, _))) if draw_status != glow::FRAMEBUFFER_COMPLETE => {
                // The view lost its drawable, or never got one. Have the next acquisition
                // attach to it again.
                log::debug!("the surface has no drawable (status {draw_status:#x}), skipping");
                self.state.attached.store(false, Ordering::Release);
                Err(crate::SurfaceError::Occluded)
            }
            Ok(Err((_, read_status))) => {
                log::error!("the swapchain framebuffer is incomplete (status {read_status:#x})");
                Err(crate::SurfaceError::Other(
                    "the swapchain framebuffer is incomplete",
                ))
            }
            Err(e) => {
                log::error!("unable to make the surface CGL context current: {e}");
                Err(crate::SurfaceError::Other(
                    "unable to make the surface CGL context current",
                ))
            }
        }
    }

    pub fn supports_srgb(&self) -> bool {
        // `GL_FRAMEBUFFER_SRGB` is core in every context this backend creates. See `present`
        // for a caveat about presenting sRGB formats.
        true
    }
}

impl crate::Surface for Surface {
    type A = super::Api;

    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        // Remove the old configuration.
        unsafe { self.unconfigure(device) };

        let swap_interval: c_int = match config.present_mode {
            wgt::PresentMode::Immediate => 0,
            wgt::PresentMode::Fifo => 1,
            _ => {
                log::error!("unsupported present mode: {:?}", config.present_mode);
                return Err(crate::SurfaceError::Other("unsupported present mode"));
            }
        };
        let cgl = &self.state.context.cgl;
        let cgl_lock = self.state.context.lock();
        let result = cgl.check(unsafe {
            (cgl.set_parameter)(
                self.state.context.raw,
                ffi::kCGLCPSwapInterval,
                &swap_interval,
            )
        });
        drop(cgl_lock);
        if let Err(e) = result {
            log::error!("unable to set swap interval: {e}");
            return Err(crate::SurfaceError::Other("unable to set swap interval"));
        }

        // Pick up a resized view before the next present.
        self.state.queue_refresh();

        let format_desc = device.shared.describe_texture_format(config.format);
        let gl = &device.shared.context.lock();

        let renderbuffer = unsafe { gl.create_renderbuffer() }.map_err(|error| {
            log::error!("Internal swapchain renderbuffer creation failed: {error}");
            crate::DeviceError::OutOfMemory
        })?;
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer)) };
        unsafe {
            gl.renderbuffer_storage(
                glow::RENDERBUFFER,
                format_desc.internal,
                config.extent.width as _,
                config.extent.height as _,
            )
        };
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };

        let framebuffer = gl
            .with_surface_context(&self.state.context, |gl| unsafe { gl.create_framebuffer() })
            .map_err(|e| {
                log::error!("unable to make the surface CGL context current: {e}");
                crate::SurfaceError::Other("unable to make the surface CGL context current")
            })?
            .map_err(|error| {
                log::error!("Internal swapchain framebuffer creation failed: {error}");
                crate::DeviceError::OutOfMemory
            });
        let framebuffer = match framebuffer {
            Ok(framebuffer) => framebuffer,
            Err(e) => {
                unsafe { gl.delete_renderbuffer(renderbuffer) };
                return Err(e.into());
            }
        };

        self.swapchain.write().replace(Swapchain {
            renderbuffer,
            framebuffer,
            extent: config.extent,
            format: config.format,
            format_desc,
        });

        Ok(())
    }

    unsafe fn unconfigure(&self, device: &super::Device) {
        let gl = &device.shared.context.lock();
        if let Some(sc) = self.swapchain.write().take() {
            unsafe { gl.delete_renderbuffer(sc.renderbuffer) };
            if let Err(e) = gl.with_surface_context(&self.state.context, |gl| unsafe {
                gl.delete_framebuffer(sc.framebuffer)
            }) {
                log::error!("unable to delete the swapchain framebuffer: {e}");
            }
        }
    }

    unsafe fn acquire_texture(
        &self,
        _timeout_ms: Option<Duration>,
        _fence: &super::Fence,
    ) -> Result<crate::AcquiredSurfaceTexture<super::Api>, crate::SurfaceError> {
        let swapchain = self.swapchain.read();
        let sc = swapchain
            .as_ref()
            .ok_or(crate::SurfaceError::Other("surface isn't configured"))?;

        // Attaching fails while the view isn't on screen, e.g. while an application configures
        // the surface of a window it shows only afterwards. Retry until it succeeds, without
        // rendering frames that couldn't be presented.
        if !self.state.attached.load(Ordering::Acquire) {
            self.state.queue_refresh();
            if !self.state.attached.load(Ordering::Acquire) {
                return Err(crate::SurfaceError::Occluded);
            }
        }
        // Neither render nor present while the window is hidden. Besides saving the work,
        // AppKit doesn't honour the swap interval for occluded windows, so presenting would
        // spin as fast as it can instead of waiting for vsync.
        if !self.state.check_window() {
            return Err(crate::SurfaceError::Occluded);
        }

        let texture = super::Texture {
            inner: super::TextureInner::Renderbuffer {
                raw: sc.renderbuffer,
            },
            drop_guard: None,
            array_layer_count: 1,
            mip_level_count: 1,
            format: sc.format,
            format_desc: sc.format_desc.clone(),
            copy_size: crate::CopyExtent {
                width: sc.extent.width,
                height: sc.extent.height,
                depth: 1,
            },
        };
        Ok(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal: false,
        })
    }

    unsafe fn discard_texture(&self, _texture: super::Texture) {}
}
