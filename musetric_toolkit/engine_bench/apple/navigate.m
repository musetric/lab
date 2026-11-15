#import <AppKit/AppKit.h>
#import <WebKit/WebKit.h>

static WKWebView *findWebView(NSView *view) {
  if ([view isKindOfClass:[WKWebView class]]) {
    return (WKWebView *)view;
  }
  for (NSView *child in view.subviews) {
    WKWebView *found = findWebView(child);
    if (found) {
      return found;
    }
  }
  return nil;
}

static BOOL isMainWindow(NSWindow *window) {
  return ![window.title.lowercaseString containsString:@"executor"];
}

static WKWebView *mainWebView(void) {
  for (NSWindow *window in NSApp.windows) {
    WKWebView *webView = findWebView(window.contentView);
    if (webView && webView.URL.absoluteString.length > 0 && isMainWindow(window)) {
      return webView;
    }
  }
  return nil;
}

static void poll(NSString *dir) {
  NSArray *names = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:dir error:nil];
  WKWebView *webView = mainWebView();
  for (NSString *name in names) {
    if (![name hasSuffix:@".js"] || !webView) {
      continue;
    }
    NSString *path = [dir stringByAppendingPathComponent:name];
    NSString *code = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
    NSString *out = [[path stringByDeletingPathExtension] stringByAppendingPathExtension:@"json"];
    [webView evaluateJavaScript:code
              completionHandler:^(id result, NSError *error) {
                NSDictionary *body = error ? @{@"error" : error.localizedDescription ?: @"error"}
                                           : @{@"value" : [NSString stringWithFormat:@"%@", result ?: @""]};
                NSData *data = [NSJSONSerialization dataWithJSONObject:body options:0 error:nil];
                [data writeToFile:out atomically:YES];
              }];
  }
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 5), dispatch_get_main_queue(), ^{
    poll(dir);
  });
}

static void navigate(int attempt) {
  const char *target = getenv("BENCH_URL");
  if (!target) {
    return;
  }
  NSURL *url = [NSURL URLWithString:[NSString stringWithUTF8String:target]];
  for (NSWindow *window in NSApp.windows) {
    WKWebView *webView = findWebView(window.contentView);
    NSString *current = webView.URL.absoluteString ?: @"";
    if (webView && current.length > 0 && isMainWindow(window)) {
      fprintf(stderr, "navigate: %s %s -> %s\n", window.title.UTF8String, current.UTF8String, target);
      [webView loadRequest:[NSURLRequest requestWithURL:url]];
      [NSApp activateIgnoringOtherApps:YES];
      [window makeKeyAndOrderFront:nil];
      return;
    }
  }
  if (attempt < 120) {
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC / 2), dispatch_get_main_queue(), ^{
      navigate(attempt + 1);
    });
  } else {
    fprintf(stderr, "navigate: no main web view\n");
  }
}

__attribute__((constructor)) static void start(void) {
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 3 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
    navigate(0);
    const char *dir = getenv("BENCH_JS_DIR");
    if (dir) {
      [NSApp activateIgnoringOtherApps:YES];
      [mainWebView().window makeKeyAndOrderFront:nil];
      poll([NSString stringWithUTF8String:dir]);
    }
  });
}
