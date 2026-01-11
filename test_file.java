@Test
public void outerInnerErrorRace() {
    for (int i = 0; i < 500; i++) {
        List<Throwable> errors = TestHelper.trackPluginErrors();
        try {

            final PublishSubject<Integer> ps1 = PublishSubject.create();
            final PublishSubject<Integer> ps2 = PublishSubject.create();
            ps1.switchMap(new Function<Integer, ObservableSource<Integer>>() {
                @Override
                public ObservableSource<Integer> apply(Integer v) throws Exception {
                    if (v == 1) {
                        return ps2;
                    }
                    return Observable.never();
                }
            })
            .test();

            final TestException ex1 = new TestException();

            Runnable r1 = new Runnable() {
                @Override
                public void run() {
                    ps1.onError(ex1);
                }
            };
            final TestException ex2 = new TestException();
            Runnable r2 = new Runnable() {
                @Override
                public void run() {
                    ps2.onError(ex2);
                }
            };
            TestHelper.race(r1, r2);

            for (Throwable e : errors) {
                assertTrue(e.toString(), e instanceof TestException);
            }
        } finally {
            RxJavaPlugins.reset();
        }
    }
}

@Test
private MimeMultipart verifyAndExtractMimeMultipart(String subject)
        throws MessagingException, IOException, InterruptedException {
    int oldCount = 0;
    int expectedEmailCount = 1;
    // wait for the server to receive the messages
    waitForServerToReceiveEmails(expectedEmailCount);
    while (!ch.qos.logback.core.net.SMTPAppenderBase.getExecutionStatus()) {
        Thread.yield();
    }
    MimeMessage[] mma = greenMailServer.getReceivedMessages();
    assertNotNull(mma);
    assertEquals(expectedEmailCount, mma.length);
    MimeMessage mm = mma[oldCount];
    // http://jira.qos.ch/browse/LBCLASSIC-67
    assertEquals(subject, mm.getSubject());
    return (MimeMultipart) mm.getContent();
}

@Test
public void testAddition() {
    // Arrange
    Calculator calculator = new Calculator();
    int a = 5;
    int b = 3;

    // Act
    int result = calculator.add(a, b);

    // Assert
    assertEquals(8, result, "The addition result should be 8");
}