        REM   This file was produced by typing in a listing from appendix B of Hugh Burns' thesis,
        REM   <https://apps.dtic.mil/dtic/tr/fulltext/u2/a106372.pdf>

        REM   (page 194)

00010   REM   <<<   INVENTION PROGRAM:  ARISTOTLE'S TOPICS   >>>
00020   REM   <<<   AUTHOR:  HUGH BURNS   >>>
00021   REM   <<<   THIS PROGRAM MAY BE USED ONLY WITH THE AUTHOR'S PERMISSION.
00022   REM   USE WITHOUT DIRECT PERMISSION VIOLATES COPYRIGHT LAW.   >>>
00030   RANDOMIZE
00040   DIM X(38)
00050   X(R)=0
00060   DIM Z(38)
00070   Z(Q)=0
00080   E=L4=D=C=Q8=E3=0   'COUNTERS
00090   PRINT
00100   PRINT
00110   PRINT
00120   PRINT
00130   PRINT
00140   PRINT,"A COMPUTER-PROMPTED INVENTION PROGRAM:"
00150   PRINT,"-------------------------------------"
00160   PRINT
00170   PRINT,"          ARISTOTLE'S TOPICS"
00180   PRINT,"          ------------------"
00190   PRINT
00200   PRINT
00210   PRINT
00220   PRINT
00230   PRINT,"HELLO AND WELCOME!"
00240   PRINT
00250   PRINT "PLEASE TYPE IN YOUR FIRST NAME:  ";
00260   LINPUT N1$
00270   IF N1$="" THEN 260
00280   PRINT
00290   PRINT "NOW, "N1$", PLEASE TYPE IN YOUR LAST NAME:  ";
00300   LINPUT N2$
00310   IF N2$="" THEN 300
00312   IF N2$="TEST!" THEN 3330
00320   PRINT
00330   PRINT
00340   PRINT "WELL, "N1$" "N2$", I HOPE I CAN BE OF SOME ASSISTANCE"
00350   PRINT "TO YOU TODAY.  IF WE TAKE EACH OTHER SERIOUSLY, YOU'LL"
00360   PRINT "THINK ABOUT YOUR TOPIC AS YOU NEVER HAVE BEFORE."
00370   PRINT
00380   PRINT
00390   PRINT,"BEFORE WE BEGIN, "N1$",  THERE'S AN OLD"
00400   PRINT "SAYING ABOUT COMPUTER-ASSISTED INSTRUCTION,  IT GOES:"
00410   PRINT
00420   PRINT,"'GARBAGE IN, GARBAGE OUT!'"
00430   PRINT
00440   PRINT "IN OTHER WORDS, YOU AND I MUST WORK TOGETHER SO"
00450   PRINT "YOU CAN GET A GOOD START ON YOUR RESEARCH PAPER."
00460   PRINT
00470   PRINT
00480   PRINT
00490   PRINT,,"(PRESS 'RETURN' TO CONTINUE.)";
00500   LINPUT A$
00510   PRINT
00520   PRINT
00530   PRINT
00540   PRINT "WOULD YOU LIKE TO REVIEW THE DIRECTIONS AND THE COMMANDS?"

        REM   (page 195)

00550   PRINT,"(YES OR NO?)"
00560   J$="*YE*" '???
00570   GOSUB 4880
00580   IF K1=1 THEN 600
00590   GOTO 1570
00600   REM   <<<   DIRECTIONS AND COMMANDS   >>>
00610   PRINT
00620   PRINT
00630   PRINT,"DIRECTIONS:"
00640   PRINT
00650   PRINT
00660   PRINT,"1.  WHEN YOU MAKE A TYPING ERROR, "N1$", AND"
00670   PRINT,"WISH TO CORRECT IT, USE THE 'RUBOUT' OR 'RUB' KEY."
00680   PRINT,"THE 'SHIFT' MUST BE DEPRESSED WHEN YOU 'RUBOUT'."
00690   PRINT,"IT MAY LOOK A LITTLE FUNNY (LIKE WRITING BACKWARDS),"
00700   PRINT,"BUT DON'T WORRY; IT WORKS THAT WAY."
00710   PRINT
00720   PRINT
00730   PRINT,"2.  REMEMBER THAT I CAN ONLY READ ABOUT A LINE AND"
00740   PRINT,"A HALF OF INFORMATION AT ONE TIME -- ABOUT THIS MUCH:"
00750   PRINT
00760   PRINT"---------------------------------------------------------------------------------------------------------"
00770   PRINT
00780   PRINT,"HIT 'RETURN' AT THAT POINT AND I'LL GENERALLY"
00790   PRINT,"LET YOU ADD MORE INFORMATION.  IF THAT DOES NOT WORK,"
00800   PRINT,"TYPE '&&' AND I'LL SAY 'GO ON, "N1$".'"
00810   PRINT
00820   PRINT
00830   PRINT,,"(PRESS 'RETURN' TO CONTINUE.)";
00840   LINPUT A$
00850   PRINT
00860   PRINT
00870   PRINT,"3.  AFTER YOU FINISH TYPING YOUR RESPONSE, YOU MUST PRESS"
00880   PRINT,"THE 'RETURN' KEY.  WHEN YOU DO , I'LL READ YOUR"
00890   PRINT,"RESPONSE AND SAY SOMETHING BACK TO YOU."
00900   PRINT
00910   PRINT
00920   PRINT,"4.  THE MOST IMPORTANT OBJECTIVE OF THIS PROGRAM"
00930   PRINT,"IS TO GET YOU THINKING ABOUT YOUR TOPIC."
00940   PRINT
00950   PRINT,"IN ORDER TO ACHIEVE THIS OBJECTIVE,"
00960   PRINT,"YOU ARE GOING TO HAVE TO FORGET THAT I AM A MACHINE."
00970   PRINT
00980   PRINT,"PLEASE ASK QUESTIONS.  YOU'LL BE SURPRISED BY HOW MUCH"
00990   PRINT,"I KNOW (OR SO I HOPE!)  I'M NOT"
01000   PRINT,"GUARANTEEING THE TRUTH, BUT I'LL DO THE BEST I CAN."
01010   PRINT,"MY MEMORY IS STILL DEVELOPING."
01020   PRINT
01030   PRINT
01040   PRINT
01050   PRINT,,"(HIT 'RETURN' TO CONTINUE.)"
01060   PRINT
01070   PRINT
01080   LINPUT A$
01090   PRINT
01100   PRINT
01110   PRINT
01120   PRINT,"COMMANDS:"

        REM   (page 196)

01130   PRINT
01140   PRINT,"TYPE IN-->","I'LL DO THIS-->"
01150   PRINT,"----------","---------------"
01160   PRINT
01170   PRINT,"STOP!","I'LL STOP ASKING QUESTIONS AND CLOSE."
01180   PRINT
01190   PRINT,"CONTINUE!","I'LL SKIP AHEAD TO THE NEXT QUESTION."
01200   PRINT
01210   PRINT,"REPEAT!","I'LL REPEAT THE QUESTION."
01220   PRINT
01230   PRINT,"DIRECTIONS!","I'LL SHOW YOU THE DIRECTIONS AGAIN."
01240   PRINT
01250   PRINT,"CHANGE!","I'LL LET YOU CHANGE OR NARROW YOUR SUBJECT."
01260   PRINT
01270   PRINT,"?","I'LL LET YOU ASK A QUESTION."
01280   PRINT
01290   PRINT,"EXPLAIN!","I'LL EXPLAIN THE QUESTION."
01300   PRINT,,"(THIS ONE IS A LOT OF FUN, "N1$".)"
01310   PRINT
01320   PRINT,"&&","I'LL LET YOU CONTINUE WITH YOUR RESPONSE."
01330   PRINT
01340   PRINT,,"(PRESS 'RETURN' TO CONTINUE.)";
01350   LINPUT A$
01360   PRINT
01370   PRINT
01380   PRINT
01390   PRINT
01400   PRINT,"TWO LAST THINGS:"
01410   PRINT
01420   PRINT,"***  THINK OF ME AS A PERSON WHO CAN ASK A LOT OF"
01430   PRINT,"INTERESTING, THOUGHT-PROVOKING, AND WILD QUESTIONS."
01440   PRINT
01450   PRINT
01460   PRINT,"***  SCREAM FOR HELP IF I START ACTING REALLY CRAZY!!"
01470   PRINT
01480   PRINT
01490   IF D=1 THEN 1510
01500   GOTO 1570
01510   PRINT,"BACK TO THE QUESTIONS, "N1$"   -->   -->   -->"
01520   PRINT
01530   PRINT
01540   PRINT
01550   PRINT,,"BUT FIRST, IS THERE"
01560   GOTO 6050
01570   PRINT
01580   PRINT
01590   PRINT
01600   PRINT
01610   PRINT "WOULD YOU LIKE A BRIEF EXPLANATION OF HOW"
01620   PRINT "ARISTOTLE'S TOPICS HELP WRITERS WRITE?"
01630   PRINT,"(YES OR NO?)"
01640   J$="*YE*"
01650   GOSUB 4880
01660   IF K1=1 THEN 1680
01670   GOTO 1930
01680   REM   <<<   DESCRIPTION OF ARISTOTLE'S TOPICS   >>>
01690   PRINT
01700   PRINT
01710   PRINT,"I'M GLAD YOU ASKED, "N1$".  BRIEFLY, THE TWENTY-EIGHT"

        REM   (page 197)

01720   PRINT "ENTHYMEME TOPICS HELP A WRITER (OR A SPEAKER) DISCOVER"
01730   PRINT "SPECIFIC ARGUMENTS ABOUT SUBJECTS."
01740   PRINT
01750   PRINT,"IN HIS 'RHETORIC', ARISTOTLE TELLS US THAT THE AIM OR GOAL"
01760   PRINT "OF RHETORIC IS TO PERSUADE AN AUDIENCE.  REMEMBER THAT TERM --"
01770   PRINT "PERSUADE."
01780   PRINT
01790   PRINT,"ARISTOTLE BELIEVED THAT IF HIS STUDENTS IN THE"
01800   PRINT "ACADEMY KNEW AND PRACTICED USING THE TOPICS, THEY WOULD BECOME"
01810   PRINT "EFFECTIVE 'PERSUADERS.'"
01820   PRINT
01830   PRINT,"YOU'LL RECOGNIZE AMONG THE TOPICS:"
01840   PRINT
01850   PRINT,"1.  QUESTIONS OF DEFINITION;"
01860   PRINT,"2.  QUESTIONS ABOUT CAUSES AND EFFECTS;"
01870   PRINT,"3.  QUESTIONS REGARDING OPPOSITES AND ASSOCIATIONS;"
01880   PRINT,"4.  QUESTIONS ABOUT CONSEQUENCES;"
01890   PRINT,"5.  AND QUESTIONS ABOUT MATTERS OF FACT AND OPINION."
01900   PRINT
01910   PRINT,,"(HIT 'RETURN' TO CONTINUE.)"
01920   LINPUT A$
01930   REM   <<<   SUBJECT SEQUENCE   >>>
01940   PRINT
01950   PRINT
01960   PRINT
01970   PRINT
01980   PRINT
01990   PRINT
02000   PRINT
02010   PRINT
02020   PRINT
02030   PRINT
02040   PRINT,"NOW I NEED TO FIND OUT WHAT YOU"
02050   PRINT "ARE WRITING ABOUT, SO WOULD YOU PLEASE TYPE IN YOUR"
02060   PRINT "SUBJECT.  I AM LOOKING FOR ONE TO THREE WORDS."
02070   PRINT
02080   PRINT
02090   PRINT
02100   PRINT
02110   PRINT
02120   PRINT
02130   PRINT,;
02140   LINPUT S$
02150   IF S$=""THEN 2140
02160   IF LEN(S$)<40 THEN 2280
02170   PRINT
02180   PRINT "THAT'S A MOUTHFUL, "N1$".  MAKE IT SHORTER, LIKE A TITLE."
02190   PRINT,"HERE ARE A FEW EXAMPLES:"
02200   PRINT
02210   PRINT,"  **   THE ENERGY CRISIS"
02220   PRINT,"  **   AUSTIN'S HISTORICAL GARDENS"
02230   PRINT,"  **   THE BERMUDA TRIANGLE"
02240   PRINT
02250   PRINT
02260   PRINT,"YOUR TURN.  WHAT IS YOUR SUBJECT?"
02270   GOTO 2120

        REM   (page 198)

02280   IF N8>0 THEN 2300
02290   GOTO 2380
02300   PRINT
02310   PRINT "YOUR REVISED SUBJECT IS "S$"."
02320   PRINT
02330   PRINT
02340   PRINT
02350   PRINT
02360   PRINT
02370   GOTO 6210
02380   J=INT(3*RND+1)
02390   ON J GOTO 2400,2440,2480
02400   PRINT   'INFORMAL ACKNOWLEDGEMENT OF SUBJECT
02410   PRINT "HOLY ELECTRONICS!  THAT'S WEIRD, I USED TO DATE A COMPUTER"
02420   PRINT "INTERESTED IN "S$"."
02430   GOTO 2520
02440   PRINT
02450   PRINT "HEY, THAT'S NEAT, "N1$"!  WE'LL HAVE A GOOD TIME THINKING"
02460   PRINT "ABOUT "S$"."
02470   GOTO 2520
02480   PRINT
02490   PRINT S$", MMMMM!  WILL YOU BE AMAZED"
02500   PRINT "BY THE RECENT SCHOLARSHIP.  BE SURE TO ASK THE LIBRARIAN"
02510   PRINT "IN THE REFERENCE AREA."
02520   REM   <<<   PURPOSE SEQUENCE   >>>
02530   PRINT
02540   PRINT
02550   PRINT
02560   PRINT
02570   PRINT
02580   PRINT,"A COMMENT ABOUT PURPOSE:"
02590   PRINT
02600   PRINT
02610   PRINT
02620   PRINT,"DURING THIS EXPLORATION PROCESS,"
02630   PRINT,"YOU WILL BE ASKED TO CLARIFY THE PURPOSE OF"
02640   PRINT,"YOUR PAPER ON "S$"."
02650   PRINT
02660   PRINT
02670   PRINT,"SO NOW WOULD YOU BRIEFLY DESCRIBE WHAT THE PURPOSE"
02680   PRINT,"OF YOUR PAPER BY COMPLETING"
02690   PRINT,"THIS STATEMENT:  THE PURPOSE OF THIS PAPER IS TO. . . ."
02691   PRINT,"(LIMIT:  ONE LINE)"
02700   PRINT
02710   PRINT
02720   LINPUT P$
02730   IF P$=""THEN 2720
02740   PRINT
02741   GOSUB 3321
02750   PRINT
02760   PRINT,"FINE, "N1$", YOU AND I WILL TALK AGAIN ABOUT YOUR"
02770   PRINT,"PURPOSE."
02780   PRINT
02790   PRINT
02800   GOTO 3330
02810   PRINT   'PURPOSE SUBROUTINE AT C+1=6
02820   PRINT

        REM   (page 199)

02830   PRINT,"BEFORE WE CONTINUE, "N1$", I WANT YOU"
02840   PRINT,"TO THINK ABOUT YOUR PURPOSE ONCE AGAIN."
02850   PRINT
02860   PRINT,"YOU HAVE ALREADY TOLD ME THAT YOUR PURPOSE WAS"
02870   PRINT"TO "P$"."
02880   PRINT
02890   PRINT
02900   PRINT,"HOW WOULD YOU COMPLETE THIS STATEMENT:"
02910   PRINT
02920   PRINT,"IF NOTHING ELSE, I WANT MY READER TO UNDERSTAND. . . ."
02921   PRINT,"(ONE LINE, PLEASE)"
02930   PRINT
02940   PRINT
02950   LINPUT P1$
02960   IF P1$="" THEN 2950
02970   PRINT
02971   GOSUB 3321
02980   PRINT,"OKAY, FINE.  KEEP YOUR PURPOSE IN MIND AS WE CONTINUE."
02990   PRINT
03000   PRINT
03010   PRINT
03020   PRINT,"HERE IS YOUR NEXT QUESTION -- NUMBER"C+1"."
03030   PRINT
03040   PRINT
03050   GOTO 3530
03060   PRINT   'PURPOSE SUBROUTINE AT C+1=12
03070   IF N4>0 THEN 3000  '??? both the variable name and the destination line are unclear
03080   PRINT
03090   PRINT,"LET'S PAUSE ONCE AGAIN TO CONSIDER YOUR INTENT."
03100   PRINT
03110   PRINT,"YOUR GENERAL PURPOSE IS TO"
03120   PRINT P$
03130   PRINT
03140   PRINT,"ALSO, YOU WANT YOUR READER TO UNDERSTAND"
03150   PRINT P1$"."
03160   PRINT
03170   PRINT
03180   PRINT,"IS THERE ANYTHING ELSE YOU WISH TO SAY ABOUT PURPOSE?"
03190   PRINT,,"(YES OR NO?)"
03200   J$="*YE*"
03210   GOSUB 4880
03220   IF K1=1 THEN 3260
03230   PRINT
03240   PRINT,"FINE, "N1$", ENOUGH ABOUT PURPOSE."
03250   GOTO 3000
03260   PRINT
03270   PRINT,"GREAT, "N1$", WHAT WOULD YOU LIKE TO ADD?"
03271   PRINT,"(ONE LINE LIMIT IN EFFECT)"
03280   PRINT
03290   PRINT
03300   LINPUT P2$
03310   IF P2$=""THEN 3300
03311   GOSUB 3321
03320   GOTO 3230
03321   PRINT
03322   PRINT,"ANY MORE?"
03323   PRINT,"(IF SO, TYPE WHATEVER IT IS; IF NOT, TYPE 'NO'.)"
03324   PRINT
03325   LINPUT A$
03326   PRINT

        REM   (page 200)

03327   RETURN
03330   PRINT   'PAGING OPENING QUESTIONING SEQUENCE
03340   PRINT
03350   PRINT
03360   PRINT
03370   PRINT
03380   PRINT,"RELAX NOW, "N1$", AND ENJOY THIS BRAINSTORMING SESSION."
03390   PRINT
03400   PRINT
03410   PRINT
03420   PRINT
03430   PRINT
03440   PRINT
03450   PRINT
03460   PRINT
03470   PRINT
03480   PRINT
03490   PRINT
03500   PRINT
03510   PRINT
03520   REM   <<<   COUNTER/EXPLORATION CONTROLS   >>>
03530   C=C+1
03540   E=L4=Q8=Q6=0
03550   IF C>30 THEN 10020
03560   IF C> 5 THEN 3610'OPENS TOTAL POOL AFTER FIVE QUESTIONS
03570   Q=R=R1=INT(10*RND+1)
03580   IF Z(Q)=1 THEN 3570
03590   Z(Q)=1
03600   GOTO 3740
03610   Q=R=R1=INT(38*RND+1)
03620   IF Z(Q)=1 THEN 3610
03630   Z(Q)=1
03640   IF Q<11 THEN 3740
03650   IF Q<21 THEN 3680
03660   IF Q<31 THEN 3700
03670   IF Q<39 THEN 3720
03680   Q=Q-10
03690   GOTO 3750
03700   Q=Q-20
03710   GOTO 3760
03720   Q=Q-30
03730   GOTO 3770
03740   ON Q GOTO 3790,4090,4370,3870,3890,3920,4400,4800,4280,4030
03750   ON Q GOTO 4060,3810,4120,4150,4180,4210,4230,4250,4000,4310
03760   ON Q GOTO 4340,3840,3940,4430,4460,4490,4520,4550,4570,4600
03770   ON Q GOTO 4620,4650,4680,4710,4740,4770,3970,4840
03780   REM   <<<   QUESTION POOL FOR ARISTOTLE'S TOPICS   >>>
03790   PRINT "WHAT IS THE OPPOSITE OF "S$"?"
03800   GOTO 5050
03810   PRINT "TAKE EACH WORD OF "S$" INDIVIDUALLY."
03820   PRINT "WHAT DOES IT MEAN?  CONNOTATIONS?  DENOTATIONS?"
03830   GOTO 5050
03840   PRINT "WHAT IS THE MOST LIKELY PLACE FOR"
03850   PRINT S$" TO EXIST?"
03860   GOTO 5050
03870   PRINT "HOW DOES TIME AFFECT "S$"?"
03880   GOTO 5050
03890   PRINT "WHAT SPECIAL EXPERIENCES MADE YOU SELECT"
03900   PRINT S$" AS YOUR TOPIC?"

        REM   (page 201)

03910   GOTO 5050
03920   PRINT "DEFINE "S$"."
03930   GOTO 5050
03940   PRINT "FILL IN THE BLANK:  IF "S$","
03950   PRINT "THEN ______________________________________."
03960   GOTO 5050
03970   PRINT "FILL IN THE BLANK:  IF "S$
03980   PRINT "PLUS ______________, THEN ______________."
03990   GOTO 5050
04000   PRINT "DIVIDE "S$" INTO THREE"
04010   PRINT "SUB-TOPICS."
04020   GOTO 5050
04030   PRINT "WHAT HAS BEEN DECIDED ABOUT "S$
04040   PRINT "TO DATE."
04050   GOTO 5050
04060   PRINT "WHAT STILL MUST BE DECIDED ABOUT"
04070   PRINT S$"? DESCRIBE."
04080   GOTO 5050
04090   PRINT "WHAT ARE THE GOOD CONSEQUENCES OF"
04100   PRINT S$"?"
04110   GOTO 5050
04120   PRINT "WHAT ARE THE BAD CONSEQUENCES OF"
04130   PRINT S$"?"
04140   GOTO 5050
04150   PRINT "WHO MIGHT BELIEVE THAT THE GOOD CONSEQUENCES OF"
04160   PRINT S$" ARE BAD?"
04170   GOTO 5050
04180   PRINT "WHO WOULD YOU CONSIDER AN AUTHORITY"
04190   PRINT "ON "S$"?"
04200   GOTO 5050
04210   PRINT "WHO GIVES (AND WHO RECEIVES) "S$"?"
04220   GOTO 5050
04230   PRINT "WHAT MAKES YOU SOMETHING OF AN AUTHORITY ON "S$"?"
04240   GOTO 5050
04250   PRINT "WHAT PARTS OF "S$" SHOULD BE"
04260   PRINT "DISCUSSED SEPARATELY?"
04270   GOTO 5050
04280   PRINT "DOES PUBLIC OPINION ABOUT "S$
04290   PRINT "DIFFER FROM PRIVATE OPINION?"
04300   GOTO 5050
04310   PRINT "DO ALL ASPECTS OF "S$" MAKE"
04320   PRINT "SENSE TO YOU?  DESCRIBE THOSE THAT DO NOT."
04330   GOTO 5050
04340   PRINT "HOW DOES THE GENERAL PUBLIC FEEL"
04350   PRINT "ABOUT "S$"?"
04360   GOTO 5050
04370   PRINT "WHAT COULD BE CONSIDERED A RESULT"
04380   PRINT "OF "S$"?"
04390   GOTO 5050
04400   PRINT "WHAT COULD BE CONSIDERED A CAUSE"
04410   PRINT "OF "S$"?"
04420   GOTO 5050
04430   PRINT "ARE THE RESULTS OF "S$" USUALLY"
04440   PRINT "THE SAME?  DESCRIBE."
04450   GOTO 5050
04460   PRINT "WHAT MOTIVATES PEOPLE TOWARD OR"
04470   PRINT "AGAINST "S$"?"
04480   GOTO 5050
04490   PRINT "WHAT WILL MAKE PEOPLE CHANGE THEIR MINDS ABOUT"

        REM   (page 202)

04500   PRINT S$"?"
04510   GOTO 5050
04520   PRINT "ARE THE CAUSES OF "S$" ALWAYS"
04530   PRINT "THE SAME?  DESCRIBE."
04540   GOTO 5050
04550   PRINT "WHAT'S INCREDIBLE ABOUT "S$"?"
04560   GOTO 5050
04570   PRINT "ARE THE CAUSES OF "S$" ALWAYS"
04580   PRINT "DIFFERENT?  EXPLAIN."
04590   GOTO 5050
04600   PRINT "WHAT CONTRADICTIONS EXIST IN "S$"?"
04610   GOTO 5050
04620   PRINT "WHAT FACTS ARE YOU UNLIKELY TO KNOW"
04630   PRINT "ABOUT "S$"?"
04640   GOTO 5050
04650   PRINT "ARE ALL THE FACTS ABOUT "S$" AS"
04660   PRINT "CLEAR AS YOU WOULD LIKE?  DESCRIBE THE AMBIGUITIES."
04670   GOTO 5050
04680   PRINT "WHAT IS A 'BETTER COURSE' FOR"
04690   PRINT S$" TO TAKE?  RECOMMENDATIONS?"
04700   GOTO 5050
04710   PRINT "WHAT WOULD BE THE WORST THING THAT COULD HAPPEN TO"
04720   PRINT S$"?"
04730   GOTO 5050
04740   PRINT "WHAT WOULD BE THE BEST THING THAT COULD HAPPEN TO"
04750   PRINT S$"?"
04760   GOTO 5050
04770   PRINT "WHAT ARE SOME OF THE PREVIOUS MISTAKES ABOUT"
04780   PRINT S$"?"
04790   GOTO 5050
04800   PRINT "WHAT OBJECTS DO YOU ASSOCIATE"
04810   PRINT "WITH "S$"?  HOW MIGHT THEY"
04820   PRINT "BE INCLUDED IN YOUR THEME?"
04830   GOTO 5050
04840   PRINT "WHAT'S INCONSISTENT ABOUT "S$"?"
04850    PRINT "PLACES?  PEOPLE?  ACTIONS?  PURPOSES?"
04860   GOTO 5050
04870   REM   <<<   KEYWORD SUBROUTINE   >>>
04880   LINPUT I$
04890   IF I$=""THEN 4880
04900   W=1
04910   K1=1
04920   I=2
04930   L0=LEN(J$)
04940   Y=INSTR(I,J$,"*")
04950   T1$=MID$(J$,I,Y-I)
04960   Y1=INSTR(W,I$,T1$)
04970   IF Y1<>0 THEN 5000
04980   K1=0
04990   RETURN
05000   I=Y+1
05010   W=Y1+1
05020   IF Y<L0 THEN 4940
05030   RETURN
05040   REM   <<<   SIGNAL REMARKS (SEMANTIC STABS) FOR BRANCHING   >>>
05050   PRINT
05060   PRINT
05070   J$="*CONTINUE!*"
05080   GOSUB 4880
05090   IF K1=1 THEN 6180

        REM   (page 203)

05092   IF I$="NO" THEN 6622
05100   J$="*STOP!*"
05110   GOSUB 4890
05120   IF K1=1 THEN 10020
05130   J$="*REPEAT!*"
05140   GOSUB 4890
05150   IF K1=1 THEN 7420
05160   IF I$="?" THEN 6750
05170   J$="*DIRECTIONS!*"
05180   GOSUB 4890
05190   D=1
05200   IF K1=1 THEN 600
05210   J$="*HOW*?*"
05220   GOSUB 4890
05230   IF K1=1 THEN 6810
05240   J$="*WHY*?*"
05250   GOSUB 4890
05260   IF K1=1 THEN 6880
05270   J$="*&&*"
05280   GOSUB 4890
05290   IF K1=1 THEN 6720
05300   J$="*EXPLAIN!*"
05310   GOSUB 4890
05320   IF K1=1 THEN 7470
05330   J$="* DO*N*T *UNDERST*"
05340   GOSUB 4890
05350   IF K1=1 THEN 7470
05360   J$="* DO*N*T *KNOW*"
05370   GOSUB 4890
05380   IF K1=1 THEN 7470
05390   J$="*CHANGE!*"
05400   GOSUB 4890
05410   IF K1=1 THEN 6920
05420   J$="*WHAT*?*"
05430   GOSUB 4890
05440   IF K1=1 THEN 7470
05450   J$="*MEAN*?*"
05460   GOSUB 4890
05470   IF K1=1 THEN 7470
05480   J$="* OR *?*"
05490   GOSUB 4890
05500   IF K1=1 THEN 7000 '???
05510   J$="*CAN I *?*"
05520   GOSUB 4890
05530   IF K1=1 THEN 7040
05540   J$="*IS *IT *?*"
05550   GOSUB 4890
05560   IF K1=1 THEN 7040
05570   J$="*BECAUSE*"
05580   GOSUB 4890
05590   IF K1=1 THEN 7080
05600   J$="*?*"
05610   GOSUB 4890
05620   IF K1=1 THEN 7110
05630   IF L4=1 THEN 6160'COUNTER TO CONTINUE AUTOMATICALLY
05635   IF Q6>0 THEN 5780   'PREVENTS SHORT RESPONSES AFTER && COMMAND
05640   IF LEN(I$)<10 THEN 7240
05650   A=LEN(I$)   'CHECKS LENGTH OF INDIVIDUAL WORDS IN STRING
05660   FOR K=1 TO A-1
05670   IF MID$(I$,K,1)=" " THEN 5710

        REM   (page 204)

05680   X=X+1
05690   IF X>15 THEN 5740
05700   GOTO 5720
05710   X=0
05720   NEXT K
05730   GOTO 5760
05740   X=0
05750   GOTO 6630
05760   X=0
05770   REM   <<<   EXPLORATION BRANCHING AND FEEDBACK   >>>
05780   PRINT
05790   PRINT
05800   F1=INT(4*RND+1)
05810   F2=INT(5*RND+1)
05820   E=E+1
05830   IF E>1 THEN 5930
05840   ON F1 GOTO 5850,5870,5890,5910
05850   PRINT "GOOD, "N1$", ADD TO YOUR RESPONSE NOW."
05860   GOTO 5050
05870   PRINT "FINE, "N1$".  WRITE SOME MORE."
05880   GOTO 5050
05890   PRINT "THAT'S THE IDEA, "N1$".  GIVE ME SOME MORE INFO NOW."
05900   GOTO 5050
05910   PRINT "BY GEORGE, "N1$", GOOD ONE.  WRITE A LITTLE MORE PLEASE."
05920   GOTO 5050
05930   ON F2 GOTO 5940,5960,5980,6000,6020
05940   PRINT "SUPER, "N1$"!"
05950   GOTO 6030
05960   PRINT "OUTSTANDING, "N1$"!"
05970   GOTO 6030
05980   PRINT "FANTASTIC, "N1$"!"
05990   GOTO 6030
06000   PRINT "TERRIFIC, "N1$"!"
06010   GOTO 6030
06020   PRINT "GREAT, "N1$"!"
06030   PRINT
06040   E3=E3+1   'E3=COUNTER FOR FULLY EXPLORED QUESTIONS
06050   PRINT,,"ANYTHING ELSE?"
06060   IF E3>2 THEN 6110
06070   PRINT,,"(YOU CAN ADD MORE INFO, ASK A"
06080   PRINT,,"QUESTION, OR GIVE A COMMAND --"
06090   PRINT,,"WHATEVER YOU WISH.)"
06100   PRINT
06110   J$="*YE*"
06120   GOSUB 4880
06130   IF K1=1 THEN 6780
06140   L4=1
06150   GOTO 5100  '??? to dodge "NO" but it also dodges "CONTINUE!"
06160   PRINT
06170   PRINT,"OKAY."
06180   PRINT
06190   IF C+1=3 THEN 7290
06200   IF C+1=8 THEN 7290
06210   IF C+1=6 THEN 2810
06220   IF C+1=12 THEN 3060
06230   PRINT
06240   PRINT
06250   H8=INT(10*RND+1)
06260   ON H8 GOTO 6270,6290,6310,6330,6350,6370,6390,6410,6430,6450

        REM   (page 205)

06270   PRINT "(SEE IF YOU CAN USE SOME MORE ACTION VERBS IN YOUR RESPONSE.)"
06280   GOTO 6460
06290   PRINT "(REMEMBER NOT TO WORRY ABOUT SPELLING!!)"
06300   GOTO 6460
06310   PRINT "(I'LL EXPLAIN MORE IF YOU ASK ME ON THIS NEXT QUESTION.)"
06320   GOTO 6460
06330   PRINT "(AFTER I ASK THIS NEXT QUESTION, TYPE 'WHAT?' AND STAND BACK.)"
06340   GOTO 6460
06350   PRINT "(SEE IF YOU CAN USE THE WORD 'BECAUSE' IN YOUR NEXT ANSWER.)"
06360   GOTO 6460
06370   PRINT "(IF YOU DON'T UNDERSTAND, JUST SAY SO NEXT TIME.  I'LL HELP.)"
06380   GOTO 6460
06390   PRINT "(I REPEAT QUESTIONS IF YOU TYPE 'REPEAT!')"
06400   GOTO 6460
06410   PRINT "(IF YOU NEED MORE ROOM, TYPE '&&' AT THE END OF A LINE.)"
06420   GOTO 6460
06430   PRINT "(TRY USING SOME MORE VERBS FOR BETTER EXPLANATIONS.)"
06440   GOTO 6460
06450   PRINT "(TRY EXPLAINING A LITTLE MORE.  LESS PHRASES, MORE SENTENCES.)"
06460   PRINT
06470   PRINT
06480   PRINT
06490   PRINT
06500   C8=INT(5*RND+1)
06510   ON C8 GOTO 6520,6540,6560,6580,6600
06520   PRINT "WE'RE MOVING RIGHT ALONG.  HERE IS QUESTION"C+1"."
06530   GOTO 6610
06540   PRINT "AND HERE COMES A REALLY INTERESTING QUESTION -- NUMBER"C+1"."
06550   GOTO 6610
06560   PRINT "QUESTION"C+1"-- ONE OF MY ALL-TIME FAVORITES COMING UP."
06570   GOTO 6610
06580   PRINT "YOUR NEXT QUESTION IS NUMBER"C+1"."
06590   GOTO 6610
06600   PRINT "HERE IS QUESTION"C+1", "N1$"."
06610   PRINT
06620   GOTO 3530
06622   PRINT   'RESPONDS TO I$=NO AFTER INVENTION PROMPTER
06623   PRINT,"YOU COULD TELL ME 'WHY NOT', BUT YOU"
06624   PRINT "MAY JUST WANT TO CONTINUE.  IF SO, TYPE 'CONTINUE!"
06625   PRINT "(DON'T FORGET THE EXCLAMATION POINT!)"
06626   GOTO 5050
06630   PRINT   'RESPONSE TO 'GARBAGE' OR JARGON
06640   PRINT,"HEY, "N1$", WHAT KIND OF LANGUAGE IS THAT?"
06650   PRINT,"TRY AGAIN. I JUST CAN'T UNDERSTAND WHAT YOU SAID?"
06660   PRINT
06670   PRINT,"(YOU MAY HAVE RUN SOME OF YOUR WORDS TOGETHER,"
06680   PRINT,"SO IF YOU CAN UNDERSTAND WHAT YOU MEAN, JUST"
06690   PRINT,"KEEP ON ANSWERING THE QUESTION.  I'LL REPEAT"
06700   PRINT,"THE QUESTION IF YOU TYPE 'REPEAT!')"
06710   GOTO 5050
06720   PRINT   'ANSWERES THE COMMAND *&&*
06730   PRINT "GO ON, "N1$"."
06735   Q6=Q6+1
06740   GOTO 5050

        REM   (page 206)

06750   PRINT   'ANSWERS THE SINGLE QUESTION MARK (I$="?")
06760   PRINT "GO AHEAD, "N1$", ASK.  I'LL DO THE BEST I CAN."
06770   GOTO 5050
06780   PRINT   'ANSWERS A *YE* TO ANYTHING ELSE?
06790   PRINT "WHAT?"
06800   GOTO 5050
06810   PRINT   'ANSWERS THE QUESTION *HOW*?*
06820   PRINT "I COULD SAY THAT THAT'S FOR ME TO KNOW AND FOR YOU TO FIND OUT."
06830   PRINT
06840   PRINT "SERIOUSLY, I CANNOT PRETEND TO KNOW 'HOW', BUT YOU"
06850   PRINT "SHOULD KEEP EXPLORING FOR AN ANSWER."
06860   PRINT
06870   GOTO 5050
06880   PRINT   'ANSWERS THE QUESTION *WHY*?*
06890   PRINT "WELL, WHY NOT?  REMEMBER WE ARE EXPLORING, BRAINSTORMING!"
06900   PRINT
06910   GOTO 5050
06920   N8=N8+1   'ANSWERS THE *CHANGE!* COMMAND
06930   IF N8>1 THEN 6970
06940   PRINT
06950   PRINT "GOOD FOR YOU, "N1$".  NOT EVERY WRITER NARROWS OR"
06960   PRINT "CHANGES HIS OR HER TOPIC THIS EARLY IN THE INVENTION PROCESS."
06970   PRINT
06980   PRINT "PLEASE TYPE IN YOUR NEW SUBJECT:"
06990   GOTO 2120
07000   PRINT   'ANSWERS QUESTION * OR *?*
07010   PRINT "WHATEVER YOU THINK BEST, "N1$".  YOU DECIDE."
07020   PRINT
07030   GOTO 5050
07040   PRINT   'ANSWERS QUESTION *CAN I *?*
07050   PRINT "YES, OF COURSE."
07060   PRINT
07070   GOTO 5050
07080   PRINT   'RESPONDS TO SUBORDINATE *BECAUSE*
07090   PRINT,"I LIKE YOUR REASONING."
07100   GOTO 5800   '???
07110   PRINT   'RESPONDS TO *?*
07120   Q8=Q8+1
07130   IF Q8<2 THEN 7180
07140   IF Q8>2 THEN 7710  '??? Looks like a typo in the original; should probably be 7210.
07150   PRINT "ANOTHER INTERESTING QUESTION.  I'D SAY 'YES'."
07160   PRINT
07170   GOTO 9910
07180   PRINT "YES, THAT SEEMS OKAY."
07190   PRINT
07200   GOTO 9970
07210   PRINT "THIS QUESTION MAY BE BETTER ANSWERED"
07220   PRINT "DURING THE RESEARCH PHASE.  KEEP IT IN MIND."
07230   GOTO 9930
07240   PRINT   'RESPONDS TO SHORT ANSWERS
07250   PRINT,"AHHH, SHORT AND SWEET.  NOW TELL ME"
07260   PRINT,"WHY?  IN OTHER WORDS, ELABORATE A LITTLE."
07270   PRINT
07280   GOTO 5050
07290   PRINT   'AUTO NARROW/CHANGE LOOP
07300   PRINT

        REM   (page 207)

07310   PRINT "DO YOU WISH TO NARROW OR CHANGE YOUR SUBJECT?"
07320   PRINT "(MAYBE REVISE THE WAY IT SOUNDS IN THESE QUESTIONS?)"
07330   PRINT,"(YES OR NO?)"
07340   J$="*YE*"
07350   GOSUB 4880
07360   IF K1=1 THEN 6920
07370   PRINT
07380   PRINT
07390   PRINT
07400   PRINT
07410   GOTO 6210
07420   PRINT   'REPRINTS QUESTION
07430   IF Q=R THEN 3640
07440   IF Q+10=R THEN 3750
07450   IF Q+20=R THEN 3760
07460   IF Q+30=R THEN 3770
07470   REM   <<<   CLARIFICATION ARRAY AND EXAMPLE SEQUENCE   >>>
07480   PRINT
07490   IF X(R)=1 THEN 9990  '???
07500   X(R)=1
07510   IF R<11 THEN 7610
07520   IF R<21 THEN 7550
07530   IF R<31 THEN 7570
07540   IF R<39 THEN 7590
07550   R1=R1-10   '???
07560   GOTO 7620
07570   R1=R1-20   '???
07580   GOTO 7630
07590   R1=R1-30   '???
07600   GOTO 7640
07610   ON R1 GOTO 7650,8320,8900,7830,7900,7960,8990,9770,8700,8190
07620   ON R1 GOTO 8260,7750,8410,8480,8530,8570,8620,8670,8120,8780
07630   ON R1 GOTO 8840,7800,8000,9070,9130,9160,9230,9280,9310,9370
07640   ON R1 GOTO 9440,9520,9560,9630,9670,9720,8050,9830
07650   PRINT "SOMETIMES A GOOD WAY TO DESCRIBE SOMETHING IS BY TELLING"
07660   PRINT "WHAT IT IS NOT.  THERE MAY OR MAY NOT BE A DIRECT"
07670   PRINT "OPPOSITE OF "S$", BUT"
07680   PRINT "SEE IF YOU CAN THINK OF ONE."
07690   PRINT
07700   PRINT "FOR EXAMPLE, IF I WERE WRITING A PAPER ON SOLAR"
07710   IF Q8=4 THEN 9990
07720   PRINT "ENERGY, AN ANSWER TO THIS QUESTION MIGHT PRODUCE A"
07730   PRINT "LIST OF EARTH'S NATURAL ENERGY RESOURCES."
07740   GOTO 9900
07750   PRINT "A 'CONNOTATION' IS AN ASSOCIATION; A 'DETONATION' IS"
07760   PRINT "A DICTIONARY MEANING.  THIS TACTIC OF THINKING ABOUT"
07770   PRINT "THE INDIVIDUAL WORDS IN A TOPIC OFTEN BRINGS"
07780   PRINT "A FRESH INSIGHT."
07790   GOTO 9930
07800   PRINT "WHERE SHOULD I GO TO SEE "S$"?"
07810   PRINT "CAN I GO INSIDE?  CAN I GOT OUTSIDE?  WHY OR WHY NOT?"
07820   GOTO 9960
07830   PRINT "ARISTOTLE THOUGHT ABOUT TIME AND CHANGE OFTEN.  DOES"
07840   PRINT S$" CHANGE OVER TIME?"
07850   PRINT
07860   PRINT "FOR EXAMPLE, IF I WERE WRITING A PAPER ABOUT DIAMOND MINING,"
07870   PRINT "I MIGHT WANT TO RESEARCH HOW TECHNOLOGY HAS CHANGED THE"
07880   PRINT "MINING PROCESS."

        REM   (page 208)

07890   GOTO 9900
07900   PRINT "IF YOU HAVE A GOOD ANSWER HERE, YOU WILL PROBABLY WRITE"
07910   PRINT "A DECENT PAPER.  BY 'SPECIAL', I MEAN 'UNIQUE',"
07920   PRINT "'INTERESTING', OR 'IMPORTANT'.  THESE EXPERIENCES DO NOT"
07930   PRINT "NECESSARILY HAVE TO BE YOURS; YOU COULD PRETEND TO BE A"
07940   PRINT "REPORTER."
07950   GOTO 9930
07960   PRINT "YOU MIGHT SPEND ALL DAY ON THIS QUESTION, BUT I AM"
07970   PRINT "AFTER A SHORT DEFINITION.  IN LESS THAN TWENTY WORDS,"
07980   PRINT "WHAT IS "S$"?"
07990   GOTO 9960
08000   PRINT "THIS IS A TYPE OF INDUCTION, "N1$".  I AM NOT TRYING"
08010   PRINT "TO BE TRICKY.  IN OTHER WORDS, IF YOUR TOPIC EXISTS,"
08020   PRINT "THEN OTHER THINGS--FEELINGS, ACTIONS, ETC.--ALSO EXISTS."
08030   PRINT "TRY MAKING A CONNECTION OR TWO."
08040   GOTO 9900
08050   PRINT "THIS QUESTION ASKS YOU TO CREATE A COMPLICATED"
08060   PRINT "INDUCTION.  THINK OF IT IN MATHEMATICAL TERMS:"
08070   PRINT
08080   PRINT,"IF 2 + ? THEN ?"
08090   PRINT
08100   PRINT "THERE ARE MANY ANSWERS (2+2=4, 2+90=92,....)."
08110   GOTO 9930
08120   PRINT "I LIKE ASKING THIS QUESTION BECAUSE IT MAY HELP YOU ORGANIZE"
08130   PRINT "YOUR PAPER.  WHAT ARE THREE OF THE MAJOR PARTS THAT CREATE"
08140   PRINT "THE WHOLE OF "S$"?"
08150   PRINT
08160   PRINT "YOU MIGHT WANT TO WRITE SOMETHING HERE ABOUT HOW THESE"
08170   PRINT "PARTS ARE RELATED."
08180   GOTO 9960
08190   PRINT "DECISIONS HAVE BEEN MADE ABOUT "S$"."
08200   PRINT "WHAT WERE THEY ABOUT?  WHO MADE THEM?"
08210   PRINT
08220   PRINT "FOR EXAMPLE, IF I WERE WRITING A PAPER ABOUT INFLATION,"
08230   PRINT "I WOULD WANT TO WRITE A PARAGRAPH OR TWO ABOUT THE"
08240   PRINT "GOVERNMENT'S LEGISLATION TO DATE."
08250   GOTO 9900
08260   PRINT "WHAT DECISIONS WILL HAVE TO BE MADE IN THE FUTURE"
08270   PRINT "CONCERNING "S$"."
08280   PRINT
08290   PRINT "FILL IN THE BLANKS:  CONCERNING "S$","
08300   PRINT "WE MUST DECIDE WHETHER OR NOT TO DO _________________________."
08310   GOTO 9930
08320   PRINT "WHAT GOOD WILL COME ABOUT FROM MANKIND'S CONCERN ABOUT"
08330   PRINT S$"?"
08340   PRINT
08350   PRINT"FOR EXAMPLE, IF I WERE WRITING A PAPER ABOUT COLLEGE"
08360   PRINT "ACADEMICS, SOME OF THE GOOD CONSEQUENCES MAY BE A BETTER"
08370   PRINT "JOB IN THE FUTURE, A FULLER UNDERSTANDING"
08380   PRINT "ABOUT OUR WORLD, AND AN APPRECIATION FOR GOOD STUDY HABITS."
08390   PRINT "(STOP THE SNICKERING AND GET ON WITH AN ANSWER.)"
08400   GOTO 9960
08410   PRINT "WHAT BAD WILL COME ABOUT FROM MANKIND'S CONCERN ABOUT"
08420   PRINT S$"?"
08430   PRINT

        REM   (page 209)

08440   PRINT "IN OTHER WORDS, WHAT WAS, IS, AND WILL BE THE 'BAD NEWS'"
08450   PRINT "OF THIS TOPIC.  IF YOU CANNOT THINK OF ANYTHING BAD, THEN"
08460   PRINT "WHY NOT?"
08470   GOTO 9900
08480   PRINT "HERE, "N1$", WE ARE SEARCHING FOR THE PEOPLE WHO"
08490   PRINT "HAVE COUNTER-ARGUMENTS.  LAWYERS ARE ALWAYS INTERESTED"
08500   PRINT "IN THIS PARTICULAR QUESTION.  MOST ISSUES WE WRITE ABOUT"
08510   PRINT "ARE NOT THAT CLEAR-CUT, NOT THAT 'BLACK AND WHITE.'"
08520   GOTO 9930
08530   PRINT "BY 'AUTHORITY', I MEAN A SO-CALLED EXPERT."
08540   PRINT "AS YOU WRITE THE PAPER, YOU MAY QUOTE THESE PEOPLE."
08550   PRINT "GENERALLY, THEIR OPINIONS ARE RESPECTED--IF NOT BELIEVED."
08560   GOTO 9960
08570   PRINT "I AM OFTEN SURPRISED BY THE CREATIVE ANSWERS TO THIS"
08580   PRINT "QUESTION.  THERE IS USUALLY AN INSIGHT IN UNDERSTANDING"
08590   PRINT "THESE ROLES.  BY 'GIVES', I MEAN 'IS RESPONSIBLE FOR'."
08600   PRINT "BY 'RECEIVES', I MEAN 'ACCEPTING THE CONSSEQUENCES OF'."
08610   GOTO 9960
08620   PRINT "YOU PROBABLY DON'T THINK OF YOURSELF AS AN AUTHORITY."
08630   PRINT "SO PRETEND THAT YOU ARE.  WHAT CREDENTIALS DO YOU THINK AN"
08640   PRINT "AUTHORITY ON "S$" SHOULD HAVE?"
08650   PRINT "EDUCATION?  POWER?  WEALTH?  COURAGE?  HUMILITY?"
08660   GOTO 9900
08670   PRINT "BEFORE SOMEONE CAN UNDERSTAND "S$","
08680   PRINT "WHAT MATTERS MUST BE UNDERSTOOD BY THEMSELVES."
08690   GOTO 9930
08700   PRINT "BY 'PUBLIC OPINION', I MEAN THE POPULAR POINT OF VIEW."
08710   PRINT "BY 'PRIVATE OPINION', I MEAN THE WAY PEOPLE ACTUALLY BEHAVE."
08720   PRINT "SOMETIMES, SUCH IRONIC DIFFERENCES HIGHLIGHT THE OLD ADAGE:"
08730   PRINT "'DO WHAT I SAY, NOT WHAT I DO!'"
08740   PRINT
08750   PRINT "FOR EXAMPLE, MANY FREE AND LIBERAL THINKERS MAY BE MORE"
08760   PRINT "CONSERVATIVE IN MAKING POLITICAL DECISIONS."
08770   GOTO 9960
08780   PRINT "THIS QUESTION IS INTENDED TO FIND OUT WHAT YOU DO NOT"
08790   PRINT "KNOW ABOUT "S$"."
08800   PRINT
08810   PRINT "SO, MAKE A LIST OF THOSE THINGS THAT ARE UNCLEAR -- THE"
08820   PRINT "BEST WAY TO NEW INSIGHTS."
08830   GOTO 9900
08840   PRINT "WHAT ARE THE MOST POPULAR OPINIONS REGARDING"
08850   PRINT S$"?"
08860   PRINT
08870   PRINT "IF THERE WERE AN ELECTION ABOUT THIS TOPIC SOMEHOW,"
08880   PRINT "HOW WOULD THE VOTERS RESPOND?  PRO?  CON?  WHY?"
08890   GOTO 9930
08900   PRINT "THIS QUESTION IS ABOUT CAUSES AND EFFECTS, BUT YOUR ANSWER"
08910   PRINT "SHOULD JUST MENTION THE EFFECTS, THE RESULTS, THE"
08920   PRINT "OUTCOMES OF "S$"."
08930   PRINT
08940   PRINT "FOR EXAMPLE, IF I WERE WRITING A PAPER ABOUT EXERCISE."
08950   PRINT "I WOULD WRITE ABOUT A STRONGER HEART, A NEWFOUND"
08960   PRINT "ALERTNESS, AND ANOTHER WAY TO SPEND MONEY (JOGGING SHOES,"

        REM   (page 210)

08970   PRINT "TENNIS RACKETS, BICYCLES, WEIGHTS, ETC.)"
08980   GOTO 9960
08990   PRINT "THIS QUESTION IS ABOUT CAUSES AND EFFECTS, BUT YOUR ANSWER"
09000   PRINT "SHOULD JUST MENTION THE CAUSES, THE REASONS,"
09010   PRINT "THE 'WHYS' REGARDING "S$"."
09020   PRINT
09030   PRINT "FOR EXAMPLE, IF I WERE WRITING ABOUT HUMAN RIGHTS PROGRAMS,"
09040   PRINT "I WOULD WRITE SOMETHING ABOUT THE"
09050   PRINT "OUTRAGES OF RACISM OUR WORLD WAS WITNESSED."
09060   GOTO 9900
09070   PRINT "BY 'RESULTS', I MEAN THE 'EFFECTS',  YOU MAY HAVE TO DIG"
09080   PRINT "UP A LITTLE HISTORY TO ANSWER THIS QUESTION, OR YOU MAY"
09090   PRINT "HAVE TO PREDICT THE FUTURE.  IN OTHER WORDS, CAN THE"
09100   PRINT "FINAL OUTCOMES OF THIS TOPIC BE PREDICTED OVER AND OVER"
09110   PRINT "AGAIN?"
09120   GOTO 9930
09130   PRINT "SIMPLY, WHAT MAKES PEOPLE FEEL THE WAY THEY DO?"
09140   PRINT "MORAL COMMITMENT?  PLEASURE?  FEAR?  PEER PRESSURE?  ETC."
09150   GOTO 9960
09160   PRINT "WHAT WOULD IT TAKE FOR MOST PEOPLE TO CHANGE THEIR MINDS"
09170   PRINT "ABOUT "S$"?"
09180   PRINT
09190   PRINT "MOST OF THE ANSWERS TO THIS QUESTION HAVE SOMETHING TO DO"
09200   PRINT "WITH A PERSON'S DIRECT INVOLVEMENT WITH A SUBJECT LIKE"
09210   PRINT "YOURS, "S$"."
09220   GOTO 9900
09230   PRINT "ARE THE ROOTS OF "S$", FIGURATIVELY"
09240   PRINT "SPEAKING, ALWAYS THE SAME?  LOOKING AT THIS MATTER"
09250   PRINT "ANOTHER WAY:  COULD YOU DESCRIBE DIFFERENT EARLY"
09260   PRINT "SYMPTOMS?  OR IS THERE JUST ONE SYMPTOM?"
09270   GOTO 9930
09280   PRINT "BY 'INCREDIBLE', I MEAN 'UNBELIEVABLE', 'AMAZING',"
09290   PRINT "'BEYOND HUMAN UNDERSTANDING', 'STRANGER THAN FICTION'."
09300   GOTO 9960
09310   PRINT "WHAT ARE SOME OF THE DIFFERENT EXPLANATIONS FOR THE"
09320   PRINT "EXISTENCE OF "S$"?"
09330   PRINT
09340   PRINT "IF THERE ARE NONE, WHY?  IS THERE"
09350   PRINT "REALLY THAT MUCH AGREEMENT?"
09360   GOTO 9900
09370   PRINT "BY 'CONTRADICTIONS', I MEAN 'THOSE MATTERS WHICH DO NOT"
09380   PRINT "BELONG TOGETHER' OR 'KINDS OF IRONY'."
09390   PRINT
09400   PRINT "IN OTHER WORDS, WHAT SHOULDN'T BE THERE, BUT IS?"
09410   PRINT "OR (YOU GUESSSED IT), WHAT SHOULD BE A PART OF"
09420   PRINT S$", BUT IS NOT."
09430   GOTO 9930
09440   PRINT "I BET YOU ARE SAYING TO YOURSELF, 'HOW SHOULD I KNOW?'"
09450   PRINT
09460   PRINT "WELL, IF YOU ARE GOING TO WRITE A CONVINCING PAPER ABOUT"
09470   PRINT S$", YOU MUST"
09480   PRINT "FIND OUT AS EARLY AS POSSIBLE THOSE AREAS WHICH NEED TO"
09490   PRINT "BE RESEARCHED.  RIGHT NOW, I'M ASKING YOU TO PREDICT"
09500   PRINT "WHERE YOU CAN FIND SOME MORE FACTS."
09510   GOTO 9960

        REM   (page 211)

09520   PRINT "WHAT PROBLEMS DO YOU HAVE UNDERSTANDING"
09530   PRINT S$" YOURSELF?  BY 'AMBIGUITIES', I"
09540   PRINT "MEAN THOSE MIXED FEELINGS YOU MAY HAVE ABOUT THIS TOPIC."
09550   GOTO 9900
09560   PRINT "BY 'BETTER COURSE', I MEAN FOR YOU TO SUGGEST A BETTER"
09570   PRINT "SOLUTION TO ANY PROBLEMS ASSOCIATED WITH"
09580   PRINT S$"."
09590   PRINT
09600   PRINT "IF YOU EXPECT PEOPLE TO BE CONVINCED BY YOUR ARGUMENT,"
09610   PRINT "YOU MUST OFFER THEM A SOUND SOLUTION."
09620   GOTO 9930
09630   PRINT "IF PEOPLE WERE NO LONGER CONCERNED ABOUT"
09640   PRINT S$", WOULD THAT BE"
09650   PRINT "THE WORST THING THAT COULD HAPPEN?  WHY OR WHY NOT?"
09660   GOTO 9960
09670   PRINT "IF EVERYONE IN THE WORLD WAS AS CONCERNED ABOUT"
09680   PRINT S$" AS YOU ARE,"
09690   PRINT "WOULD THAT BE THE BEST THING THAT COULD HAPPEN?"
09700   PRINT "WHY OR WHY NOT?"
09710   GOTO 9900
09720   PRINT "SIMPLY, WHAT HAS BEEN WRONG WITH THE WAY"
09730   PRINT S$" HAS BEEN HANDLED."
09740   PRINT "MAYBE 'MISTAKE' IS TOO HARSH A TERM; 'MISTREATMENT' MAY"
09750   PRINT "BE BETTER FOR THIS TOPIC."
09760   GOTO 9900
09770   PRINT "IF I SAY 'BLACK', YOU SAY 'WHITE'."
09780   PRINT "IF I SAY 'HEADACHE', YOU SAY 'ASPIRIN'."
09790   PRINT
09800   PRINT "NOW, "N1$", IF I SAY "S$","
09810   PRINT "WHAT DO YOU SAY?"
09820   GOTO 9930
09830   PRINT "BY 'INCONSISTENT', I MEAN TO SUGGEST THOSE MATTERS"
09840   PRINT "WHICH SEEM 'OUT OF PLACE.'"
09850   PRINT
09860   PRINT "'INCONSISTENT' MAY ALSO SUGGEST THAT SOME THINGS ABOUT"
09870   PRINT S$" CHANGE MORE OFTEN"
09880   PRINT "THAN OTHER THINGS.  WHAT MIGHT THEY BE?"
09890   GOTO 9960
09900   PRINT   'PROMPTERS AFTER CLARIFICATION
09910   PRINT,"TRY ANSWERING THIS QUESTION NOW."
09920   GOTO 5050
09930   PRINT
09940   PRINT,"WHAT ARE YOU THINKING NOW, "N1$"?"
09950   GOTO 5050
09960   PRINT
09970   PRINT,"YOUR TURN, "N1$"."
09980   GOTO 5050
09990   PRINT   'SECOND RESPONSE AFTER CLARIFICATION REQUEST
10000   PRINT "THAT'S ABOUT ALL I CAN ADD AT THE MOMENT.  SORRY!"
10010   GOTO 9930
10020   REM   <<<   CLOSING SEQUENCES   >>>
10030   IF C<3 THEN 10200
10040   IF C<7 THEN 10290
10050   PRINT
10060   PRINT
10070   PRINT,"YOU EXPLORED"E3"QUESTIONS OUT OF THE"C"I ASKED."
10080   PRINT,"THAT'S"(E3/C)*100"PERCENT."
10090   PRINT
10100   PRINT,"LET ME REMIND YOU THAT YOU ARE STILL IN THE FIRST STAGES"

        REM   (page 212)

10110   PRINT,"OF THE CREATIVE PROCESS.  THESE IDEAS MUST SIMMER NOW."
10120   PRINT
10130   PRINT,"ALSO, I HOPE YOU CAN CREATE SOME OF YOUR OWN 'TOPIC'"
10140   PRINT,"QUESTIONS.  I WON'T ALWAYS BE AROUND TO HELP!!!"
10150   PRINT
10160   PRINT,,"HOPE YOUR PAPER IS TERRIFIC!"
10170   PRINT
10180   PRINT,,"GOOD BYE & GOOD LUCK!"
10190   STOP
10200   PRINT
10210   PRINT
10220   PRINT,"WHY, "N1$", YOU ARE IN A HURRY TODAY."
10230   PRINT
10240   PRINT,"YOU WILL NEED TO SPEND MORE TIME THINKING ABOUT"
10250   PRINT,S$"."
10260   PRINT
10270   PRINT,"SORRY I COULD NOT HELP YOU MORE.  BYE."
10280   STOP
10290   PRINT
10300   PRINT
10310   PRINT,"YOU ARE DEFINITELY A DEEP THINKER, "N1$"."
10320   PRINT
10330   PRINT,"YOU WERE ASKED"C"QUESTIONS AND FULLY EXPLORED"
10340   PRINT,E3"OF THEM."
10350   PRINT
10360   PRINT,"PLEASE COME BACK AGAIN WHEN YOU CAN STAY LONGER."
10370   PRINT
10380   PRINT,,"GOOD-BYE."
10390   END
