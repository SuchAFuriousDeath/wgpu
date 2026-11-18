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

use alloc::{ffi::CString, format, string::String, sync::Arc, vec::Vec};
use core::{
    ffi::{c_int, c_void, CStr},
    fmt,
    mem::ManuallyDrop,
    ptr,
    time::Duration,
};

use glow::HasContext;
use objc2::{
    encode::{Encoding, RefEncode},
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
    pub type FlushDrawable = unsafe extern "C" fn(ctx: CGLContextObj) -> CGLError;
    pub type LockContext = unsafe extern "C" fn(ctx: CGLContextObj) -> CGLError;
    pub type ErrorString = unsafe extern "C" fn(error: CGLError) -> *const c_char;

    unsafe extern "C" {
        // Both are part of libSystem, which every macOS binary links.
        pub static _dispatch_main_q: c_void;
        pub fn dispatch_sync_f(
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
    flush_drawable: ffi::FlushDrawable,
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
            flush_drawable: sym(&library, c"CGLFlushDrawable")?,
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
        // than failing outright.
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
                    attributes.push(ffi::kCGLPFAAccelerated);
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
}

impl fmt::Debug for AdapterContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterContext").finish_non_exhaustive()
    }
}

#[cfg(send_sync)]
static_assertions::assert_impl_all!(AdapterContext: Send, Sync);

impl AdapterContext {
    pub fn is_owned(&self) -> bool {
        self.inner.lock().context.is_some()
    }

    /// Returns the underlying `CGLContextObj`, or null for an external adapter.
    pub fn raw_context(&self) -> *mut c_void {
        match self.inner.lock().context {
            Some(ref context) => context.raw.cast(),
            None => ptr::null_mut(),
        }
    }

    /// Obtain a lock to the CGL context and get handle to the [`glow::Context`] that can be
    /// used to do rendering.
    #[track_caller]
    pub fn lock(&self) -> AdapterContextLock<'_> {
        let inner = self
            .inner
            // Don't lock forever. If it takes longer than 1 second to get the lock we've got a
            // deadlock and should panic to show where we got stuck
            .try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
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
}

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

        let inner = self.inner.lock();
        let share = inner.context.as_ref();
        let context = CglContext::new(&self.pixel_format, share).map_err(|e| {
            crate::InstanceError::new(format!("unable to create surface CGL context: {e}"))
        })?;
        drop(inner);

        let view = handle.ns_view.as_ptr().cast();
        let ns_context = NsOpenGlContext::new(&context)?;
        ns_context.set_view(view);

        Ok(Surface {
            view,
            ns_context: ManuallyDrop::new(ns_context),
            context: ManuallyDrop::new(context),
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
                AdapterContext {
                    inner: self.inner.clone(),
                },
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
                AdapterContext {
                    inner: Arc::new(Mutex::new(Inner {
                        gl: ManuallyDrop::new(context),
                        context: None,
                    })),
                },
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

/// Runs `f` on the main thread and waits for it to finish.
///
/// AppKit requires `NSOpenGLContext`'s view management to happen on the main thread. This
/// blocks until the main thread's run loop gets to it, so calling it from another thread while
/// the main thread waits on that thread deadlocks.
fn run_on_main<F: FnOnce() + Send>(f: F) {
    if MainThreadMarker::new().is_some() {
        f();
        return;
    }

    unsafe extern "C" fn trampoline<F: FnOnce()>(context: *mut c_void) {
        let f = unsafe { &mut *context.cast::<Option<F>>() };
        if let Some(f) = f.take() {
            f();
        }
    }

    let mut f = Some(f);
    unsafe {
        ffi::dispatch_sync_f(
            ptr::addr_of!(ffi::_dispatch_main_q),
            ptr::addr_of_mut!(f).cast(),
            trampoline::<F>,
        )
    };
}

/// Lets a pointer to an Objective-C object cross into [`run_on_main`].
struct MainThreadPtr(*mut AnyObject);

unsafe impl Send for MainThreadPtr {}

/// An `NSOpenGLContext` wrapping a surface's CGL context, used only to attach it to a view.
struct NsOpenGlContext {
    raw: Retained<AnyObject>,
}

impl NsOpenGlContext {
    fn new(context: &CglContext) -> Result<Self, crate::InstanceError> {
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

        Ok(Self { raw })
    }

    fn set_view(&self, view: *mut AnyObject) {
        let context = MainThreadPtr(Retained::as_ptr(&self.raw).cast_mut());
        let view = MainThreadPtr(view);
        run_on_main(move || {
            let (context, view) = (context, view);
            autoreleasepool(|_| unsafe {
                let _: () = msg_send![context.0, setView: view.0];
            });
        });
    }

    /// Updates the drawable after the view's size or position changed.
    ///
    /// AppKit refuses to attach a context to a view that isn't on screen yet ("invalid
    /// drawable"), so this also retries attaching `view` if the context has none.
    fn update(&self, context: &CglContext, view: *mut AnyObject) {
        let ns_context = MainThreadPtr(Retained::as_ptr(&self.raw).cast_mut());
        let view = MainThreadPtr(view);
        let cgl_context = context;
        run_on_main(move || {
            let (ns_context, view) = (ns_context, view);
            let _cgl_lock = cgl_context.lock();
            autoreleasepool(|_| unsafe {
                let current_view: *mut AnyObject = msg_send![ns_context.0, view];
                if current_view.is_null() {
                    let _: () = msg_send![ns_context.0, setView: view.0];
                }
                let _: () = msg_send![ns_context.0, update];
            });
        });
    }

    fn clear_drawable(&self) {
        let context = MainThreadPtr(Retained::as_ptr(&self.raw).cast_mut());
        run_on_main(move || {
            let context = context;
            autoreleasepool(|_| unsafe {
                let _: () = msg_send![context.0, clearDrawable];
            });
        });
    }
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

pub struct Surface {
    /// The `NSView` the surface presents to. Not retained, the window must outlive the surface.
    view: *mut AnyObject,
    ns_context: ManuallyDrop<NsOpenGlContext>,
    context: ManuallyDrop<CglContext>,
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

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Drop for Surface {
    fn drop(&mut self) {
        self.ns_context.clear_drawable();
        let ns_context = unsafe { ManuallyDrop::take(&mut self.ns_context) };
        let raw = MainThreadPtr(Retained::into_raw(ns_context.raw));
        run_on_main(move || {
            let raw = raw;
            // SAFETY: Balances the `Retained` that `into_raw` gave up.
            drop(unsafe { Retained::from_raw(raw.0) });
        });
        // The `NSOpenGLContext` has released its reference to the CGL context by now.
        unsafe { ManuallyDrop::drop(&mut self.context) };
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
        let cgl = &self.context.cgl;

        let result = gl.with_surface_context(&self.context, |gl| {
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

            // `GL_FRAMEBUFFER_SRGB` is never enabled on the surface context, so the blit copies
            // the bytes without any sRGB conversion.
            //
            // Note the Y-flipping here. GL's presentation is not flipped,
            // but main rendering is. Therefore, we Y-flip the output positions
            // in the shader, and also this blit.
            unsafe {
                gl.blit_framebuffer(
                    0,
                    sc.extent.height as i32,
                    sc.extent.width as i32,
                    0,
                    0,
                    0,
                    sc.extent.width as i32,
                    sc.extent.height as i32,
                    glow::COLOR_BUFFER_BIT,
                    glow::NEAREST,
                )
            };
            unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

            cgl.check(unsafe { (cgl.flush_drawable)(self.context.raw) })
        });

        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                log::error!("unable to flush the surface drawable: {e}");
                Err(crate::SurfaceError::Other(
                    "unable to flush the surface drawable",
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
        // `GL_FRAMEBUFFER_SRGB` is core in every context this backend creates.
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
        let cgl = &self.context.cgl;
        if let Err(e) = cgl.check(unsafe {
            (cgl.set_parameter)(self.context.raw, ffi::kCGLCPSwapInterval, &swap_interval)
        }) {
            log::error!("unable to set swap interval: {e}");
            return Err(crate::SurfaceError::Other("unable to set swap interval"));
        }

        // Pick up a resized view before the next present.
        self.ns_context.update(&self.context, self.view);

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
            .with_surface_context(&self.context, |gl| unsafe { gl.create_framebuffer() })
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
            if let Err(e) = gl.with_surface_context(&self.context, |gl| unsafe {
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
