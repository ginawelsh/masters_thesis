# LLM AI detection workflows
# Detection tool: DeepSeek API
# Quiz: 1st half = bachelor's thesis abstracts (formal), 2nd half = Reddit comments (informal)

import os
import time
import json
import pandas as pd
from openai import OpenAI

MAX_RETRIES = 5
RETRY_BASE_SEC = 10

# Load .env from project root
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ---------------------------------------------------------------------------
# QUIZ DATA
# First half: bachelor's thesis abstracts (formal register)
# Second half: Reddit thread comments (informal register)
# is_human is ground truth – NOT passed to the API
# ---------------------------------------------------------------------------

ABSTRACTS_QUIZ = [
    {
        "title": "Balettmapp Kungliga Operan: Operahusens design och arkitektur",
        "text": "En presentation av Kungliga Svenska Baletten. Denna ska distribueras till operahus världen över innan ny turné påbörjas. Shanghais operahus var det första som fick den färdiga produkten. Fördjupningen rör olika operahus design och arkitektur ut- och invändigt. Även en del av husens relevanta historia tas upp.",
        "is_human": True,
    },
    {
        "title": "Metoder vid implementation av affärssystem: en studie om anpassning",
        "text": "Dagens företag är i stort behov av information. Företag som har många avdelningar behöver möjligheter för att enkelt och smidigt kunna samordna information för hela företaget. Affärssystem erbjuder denna möjlighet genom en modulbaserad konstruktion som integrerar ett företags samtliga avdelningar i ett enda system. Problem kan dock uppstå då ett valt affärssystem inte passar företagets verksamhet. En kompromiss mellan system och verksamhet måste då försöka uppnås. Ett affärssystem implementeras ofta i samband med en metod, vilken ger detaljerade beskrivningar för hur implementeringen ska gå till. På vilka grunder väljer företagen att arbeta efter en specifik metod då system och verksamhet behöver anpassas till varandra? Detta arbete bygger på en undersökning som har utförts genom en survey där fem företag har medverkat. Resultatet av undersökningen visar ett antal metoder samt presenterar orsaker till varför dessa metoder används.",
        "is_human": True,
    },
    {
        "title": "Vägen till jobb: Sambandet mellan anställningsbarhet, Big Five-personlighet och proaktivt arbetssökande – med fokus på självförtroende och anpassningsförmåga",
        "text": "Denna studie undersöker hur anställningsbarhet kan förstås i relation till individers personlighet och deras sökbeteenden på arbetsmarknaden. Med utgångspunkt i Big Five-modellen analyseras hur olika personlighetsdrag kan hänga samman med proaktiva strategier i jobbsökandet, såsom att aktivt söka information, bygga nätverk och ta egna initiativ. Vidare belyses självförtroendets betydelse som en möjlig resurs i processen, både för att våga agera och för att uthålligt hantera motgångar kopplade till att söka arbete.\n\nStudiens fokus ligger även på anpassningsbarhet som en central dimension av anställningsbarhet, där individens förmåga att justera mål, strategier och kompetenser efter arbetsmarknadens krav antas påverka möjligheterna att få och behålla ett arbete. Sammantaget syftar undersökningen till att bidra med en mer nyanserad bild av hur personliga egenskaper och beteendemönster samverkar i övergången mellan studier och arbetsliv. Genom att koppla samman personlighetsdrag, proaktivt beteende och självförtroende diskuteras vilka faktorer som kan underlätta ett framgångsrikt jobbsökande och därmed stärka individens upplevda och faktiska anställningsbarhet.",
        "is_human": False,
    },
    {
        "title": "Mer lättläst: Påbyggnad av ett automatiskt omskrivningsverktyg till lätt svenska",
        "text": "Det svenska språket ska finnas tillgängligt för alla som bor och verkar i Sverige. Därförär det viktigt att det finns lättlästa alternativ för dem som har svårighet att läsa svensktext. Detta arbete bygger vidare på att visa att det är möjligt att skapa ett automatisktomskrivningsprogram som gör texter mer lättlästa. Till grund för arbetet liggerCogFLUX som är ett verktyg för automatisk omskrivning till lätt svenska. CogFLUXinnehåller funktioner för att syntaktiskt skriva om texter till mer lättläst svenska.Omskrivningarna görs med hjälp av omskrivningsregler framtagna i ett tidigare projekt.I detta arbete implementeras ytterligare omskrivningsregler och även en ny modul förhantering av synonymer. Med dessa nya regler och modulen ska arbetet undersöka omdet är det är möjligt att skapa system som ger en mer lättläst text enligt etableradeläsbarhetsmått som LIX; OVIX och Nominalkvot. Omskrivningsreglerna ochsynonymhanteraren testas på tre olika texter med en total lägnd på ungefär hundra tusenord. Arbetet visar att det går att sänka både LIX-värdet och Nominalkvoten signifikantmed hjälp av omskrivningsregler och synonymhanterare. Arbetet visar även att det finnsfler saker kvar att göra för att framställa ett riktigt bra program för automatiskomskrivning till lätt svenska.",
        "is_human": True,
    },
    {
        "title": "Snabb art- och genusidentifiering av bakterier direkt från positiva blododlingar med MALDI\u2011TOF MS: utvärdering av NaCl\u2011metoden i BacT/ALERT vid sepsismisstanke",
        "text": "Denna studie behandlar möjligheten att snabbt art- och genusbestämma bakterier direkt från positiva blododlingar med hjälp av MALDI\u2011TOF MS, med fokus på ett kliniskt relevant flöde kopplat till blododlingssystemet BacT/ALERT. Utgångspunkten är att sepsis och andra blodbanerelaterade infektioner kräver tidig och träffsäker diagnostik, eftersom fördröjd identifiering av agens kan påverka val av antibiotikabehandling och därmed patientutfall. \n\nArbetet undersöker därför en metodik där provmaterial från blododlingar bereds för direktanalys utan att först behöva odlas ut på agar, vilket annars förlänger tiden till svar. Särskild vikt läggs vid NaCl\u2011metoden som provberedningsstrategi för att separera och koncentrera bakterier från blododlingsbuljongen på ett sätt som är kompatibelt med MALDI\u2011TOF MS. Studien relaterar metodens användbarhet till olika sepsistyper och den variation i bakterieflora som kan förekomma, med målet att bedöma om identifiering på genus- och artnivå kan erhållas tillräckligt snabbt och tillförlitligt för kliniskt beslutsstöd.\n\nSammanfattningsvis positioneras arbetet i gränslandet mellan klinisk mikrobiologi och diagnostikutveckling: genom att kombinera blododling (BacT/ALERT) med direktidentifiering via MALDI\u2011TOF MS och en förenklad provberedning (NaCl\u2011metoden) syftar studien till att bidra till kortare ledtider från positiv blododling till mikrobiologiskt svar, och därigenom till mer riktad behandling vid misstänkt sepsis.",
        "is_human": False,
    },
    {
        "title": "Kuratorers hantering av anmälningsskyldighet på ungdomsmottagningar: bedömningar av mognad och sexuella relationer i möten med sexuellt aktiva barn under 15 år",
        "text": "Denna studie behandlar kuratorers anmälningsskyldighet på ungdomsmottagningar och hur den aktualiseras i mötet med barn under 15 år som är sexuellt aktiva. Utifrån kuratorns professionella roll undersöks hur ett urval kuratorer resonerar kring bedömningar av barnets situation och vilka faktorer som påverkar beslut om att göra en orosanmälan. Särskilt fokus ligger på hur kuratorerna väger juridiska krav och verksamhetens uppdrag mot behovet av förtroendefulla samtal, integritet och stöd.\n\nI studien framträder bedömning och mognad som centrala begrepp i kuratorernas arbete. Kuratorerna behöver tolka barnets ålder i relation till upplevd mognad, grad av frivillighet, eventuella maktobalanser, och omständigheter som kan tyda på utsatthet eller exploatering. Sexuell aktivitet hos barn under 15 år beskrivs därmed inte som en fråga som automatiskt leder till anmälan, utan som något som kräver en kontextuell och individuell bedömning. Samtidigt belyses hur anmälningsskyldigheten fungerar som en tydlig yttre ram som kan skapa etiska dilemman, exempelvis när kuratorn bedömer att en anmälan riskerar att skada alliansen eller minska benägenheten att söka hjälp.\n\nSammanfattningsvis synliggör studien en praktik där kuratorer kontinuerligt balanserar stödjande samtal och skyddsperspektiv. Resultaten pekar på att bedömningar av mognad och risk blir avgörande för hur anmälningsskyldigheten förstås och tillämpas, och att kuratorernas handlingsutrymme påverkas av både juridiska tolkningar och professionella överväganden i arbetet med unga.",
        "is_human": False,
    },
    {
        "title": "Går det att påverka iranska aktörer? En fältteoretisk analys av USA:s kapacitet till strategiskt inflytande",
        "text": "Denna studie undersöker om, och i vilken utsträckning, iranska aktörer kan påverkas av USA samt vad som avgör USA:s förmåga att utöva strategiskt inflytande. Utifrån ett fältperspektiv analyseras inflytande som något som uppstår i relationer mellan aktörer inom ett specifikt politiskt och säkerhetspolitiskt \"fält\", där makt inte enbart förstås som materiella resurser utan också som positioner, normer, legitimitet och etablerade handlingsmönster.\n\nUppsatsen syftar till att belysa vilka verktyg och mekanismer USA använder för att påverka iranska aktörer, samt vilka begränsningar som finns inbyggda i fältets struktur. Genom att fokusera på hur aktörerna är placerade i fältet och vilka intressen, kapitalformer och strategier som är tillgängliga för dem, kan studien visa hur inflytande både möjliggörs och motverkas. Centralt är antagandet att iranska aktörers mottaglighet för påverkan beror på deras interna konkurrens, deras behov av legitimitet och resurser samt hur de tolkar hot, status och handlingsutrymme.\n\nSammanfattningsvis bidrar uppsatsen med en analys som problematiserar föreställningen om strategiskt inflytande som en enkel fråga om \"påtryckningar\" eller \"eftergifter\". I stället framställs USA:s möjligheter att påverka som villkorade av fältets dynamik: vilka aktörer som dominerar, vilka spelregler som råder och vilka kostnader inflytandeförsök skapar för iranska aktörer. Studien förväntas därmed ge en mer nyanserad förståelse av varför vissa amerikanska strategier kan få genomslag medan andra möter motstånd eller ger motsatt effekt.",
        "is_human": False,
    },
    {
        "title": "Upplevelser av interaktionen mellan klient och socialsekreterare -sett ur ett maktperspektiv",
        "text": "För socialtjänstens klienter innebär klientskapet en ovillkorligt underordnad roll, där individen befinner sig i en situation denne inte styr över. Individen etablerar en relation till den offentliga hjälpapparaten som samtidigt är en maktapparat. Det övergripande syftet med studien är att skapa förståelse för hur den specifika ekonomisektionen inom socialförvaltningen, i en kommun i södra Sverige, som avgränsat socialt fält, är konstruerat. Ett delsyfte är att belysa hur enskilda socialsekreterare inom denna sektion samt ett urval av deras klienter, upplever interaktionen vid handläggning av ekonomiskt bistånd. Ytterligare ett delsyfte är att få en förståelse för hur de sociala konfigurationerna ter sig samt hur makt konstrueras, i interaktionen mellan klient och socialsekreterare. Tidigare forskning inom området, av Greta-Marie Skau och Leila Billqvist, belyser problematiken i den svåra balanssituation som socialsekreterare befinner sig i, vad gäller att ha makt och ge hjälp samt vad som sker i klientprocessen. Studiens teoretiska ansats utgörs av Tillys teori om beständig ojämlikhet, där fokus ligger på kombinationer av sociala konfigurationer samt den kategoriella ojämlikhetens orsaksmekanismer. Studien metodologiska ansats är av abduktiv art, varpå Tillys teori utgör grunden för den empiriska datainsamlingen. Genom studiens resultat modifieras därefter denna teori. För den empiriska datainsamlingen tillämpas en metodtriangulering, i form av personliga dokument samt semistrukturerade intervjuer. Huvuddraget i resultatet inbegriper upplevelser av distinktionen i socialsekreterarrollen, vilken består av dels en hjälpande/behandlande funktion, dels en myndighetsutövande funktion. Resultatet påvisar även en distinktion vad gäller klientrollen, då den består av såväl rättigheter som skyldigheter. Vidare presenteras upplevelser av interaktionen mellan klient och socialsekreterare, varigenom ojämlikheter upprättas och vidmakthålls. Av analysen framgår att en beständig ojämlikhet råder i interaktionen mellan klient och socialsekreterare. Denna ojämlikhet yttrar sig inom det kategoriella paret, vilka karaktäriseras av inre och yttre kategorier, således tenderar den kategoriella ojämlikheten att vara strukturellt betingad. Ojämlikheten och maktobalansen upprättas genom exploatering och möjlighetsansamling samt vidmakthålls genom efterlikning och anpassning.",
        "is_human": True,
    },
    {
        "title": "Användarverifiering från webbkamera",
        "text": "Arbetet som presenteras i den här rapporten handlar om ansiktsigenkänning från webbkameror med hjälp av principal component analysis samt artificiella neurala nätverk av typen feedforward. Arbetet förbättrar tekniken med hjälp av filterbaserade metoder som bland annat används inom ansiktsdetektering. Dessa filter bygger på att skicka med redundant data av delregioner av ansiktet.",
        "is_human": True,
    },
    {
        "title": "Inlärningsstilar som verktyg för att individanpassa undervisningen – en studie om möjligheter och begränsningar",
        "text": "Titeln *\"Inlärningsstilar – ett sätt att individanpassa undervisningen?\"* antyder en undersökning av om och hur idén om inlärningsstilar kan användas som grund för att anpassa undervisning efter elevers individuella behov. En rimlig utgångspunkt är att skolan förväntas erbjuda en undervisning som tar hänsyn till elevers olikheter, och att begreppet inlärningsstilar ofta lyfts fram som ett praktiskt verktyg för detta. Samtidigt väcker frågetecknet i titeln frågan om begreppet verkligen är tillräckligt hållbart, både teoretiskt och empiriskt, för att fungera som ett stöd i pedagogisk planering.\n\nSammanfattningsvis kan arbetet förstås som en problematisering av relationen mellan individanpassning och kategorisering: att dela in elever i \"stilar\" kan upplevas ge läraren ett språk för variation och differentiering, men riskerar också att förenkla komplexa lärprocesser och leda till låsningar i synen på elevers förmågor. I en kandidatuppsatsstil skulle analysen troligen belysa hur inlärningsstilar definieras och används i skolpraktik, vilka argument som framförs för deras pedagogiska värde samt vilka invändningar som finns utifrån forskning om lärande. En sannolik slutsats är att individanpassning kan vinna på varierade arbetssätt och medveten didaktisk flexibilitet, men att undervisning som utformas utifrån fasta antaganden om elevers specifika \"inlärningsstil\" behöver hanteras kritiskt och inte ersätta andra, mer evidensnära sätt att förstå och stödja elevers lärande.",
        "is_human": False,
    },
    {
        "title": "Processer och informationsstöd för uppföljning av efterkalkyler i Portsystem 2000 AB:s affärssystem: från förkalkyl till kvalitetssäkrad återkoppling",
        "text": "Denna studie behandlar hur Portsystem 2000 AB arbetar med uppföljning av efterkalkyler och hur detta uppföljningsarbete påverkar företagets kalkylprocess och beslutsunderlag. Utifrån begreppen förkalkyl och efterkalkyl undersöks hur kostnader och intäkter planeras inför ett uppdrag samt hur utfallet senare sammanställs, analyseras och återförs till verksamheten. Ett särskilt fokus ligger på vilka rutiner och processer som finns för att jämföra förkalkyl med faktiskt resultat, identifiera avvikelser och använda lärdomar i kommande offerter och projekt.\n\nVidare analyseras affärssystemets och övriga informationssystems roll i uppföljningsarbetet. Studien belyser hur data samlas in, registreras och bearbetas, samt vilka möjligheter och begränsningar systemstödet innebär för att skapa en effektiv och tillförlitlig efterkalkyl. Kvalitetsaspekter, såsom datakvalitet, spårbarhet och enhetliga arbetssätt, lyfts fram som centrala för att efterkalkylerna ska kunna fungera som ett praktiskt styr- och förbättringsverktyg.\n\nSammantaget syftar arbetet till att beskriva och bedöma hur uppföljningen av efterkalkyler kan utvecklas för att stärka kopplingen mellan kalkylarbete och verksamhetsstyrning. Genom att tydliggöra processflöden, ansvarsfördelning och informationshantering skapas ett underlag för förbättringar som kan bidra till bättre precision i förkalkyler, mer systematiskt lärande och ökad kvalitet i företagets ekonomiska uppföljning.",
        "is_human": False,
    },
    {
        "title": "Pappersmakulatur vid Bobergs Tryckeri AB - orsaker och förbättringsförslag",
        "text": "Pappersmakulatur uppstår i tryckpressarna, men orsakerna finns i företagets alla funktioner. För ett framgångsrikt förbättringsarbete med att minska makulaturen krävs därför att all personal är engagerad. Examensarbetet utfördes på Bobergs Tryckeri AB i Falun, ett familjeägt företag med 60 anställda, som producerar personifierad direktreklam och blanketter. Syftet var att undersöka hur pappersmakulaturen kan minska och målet var att hitta orsaker samt ge förslag på åtgärder för att minska den. Genom intervjuer med personalen kartlades produktionssprocessen och utifrån det utarbetades förbättringsförslag. Exempel på förbättringsförslag är att kontinuerligt mäta och rapportera makulaturen, förbättra kommunikationen mellan avdelningarna och utveckla färgstyrningen. Litteraturstudier och kontakter med nyckelpersoner inom branschorganisationer och andra liknande företag, var till stor hjälp i arbetet. Fördjupningsdelen i projektet har sin grund i boken The Printer's Guide to Waste Reduction av Tim Dalton.",
        "is_human": True,
    },
    {
        "title": "Visionen om det narkotikafria samhället: En diskursanalys",
        "text": "Syftet med denna uppsats är att belysa hur visionen om det narkotikafria samhället uppenbarar sig i diskurser som behandlar narkotikamissbruk som socialt problem. Empiriskt material i form av ett policydokument och en debattföljetong har analyserats utifrån ett socialkonstruktivistiskt och diskursanalytiskt perspektiv. Analysen pekar mot att definitionskampen om narkotikamissbruk som socialt problem finns över många diskurser; samtidigt ses visionen om det narkotikafria samhället ofta vara närvarande. En nationell diskurs som kan ses utgå från vad vi som nation anser och vill ha i vårt land. En moraldiskurs som lägger värde i olika beteenden och åsikter. En juridisk och polisiär diskurs som utifrån narkotikas juridiska status ramar in fenomenet och en barn-och ungdomsdiskurs som formulerar narkotikan som det största hotet. Alla dessa kan enligt studien ses som en del i konstruktionen av narkotikamissbruk som socialt problem.",
        "is_human": True,
    },
    {
        "title": "Filmeventens framväxt i Skåne: En studie av Film i Skånes regionala eventkultur och verksamheter som M:Dox, CineSkåne och Filmbar",
        "text": "Denna studie undersöker den så kallade \"eventbubblan\" i södra Sverige genom en analys av Film i Skånes eventverksamhet. Med utgångspunkt i ett regionalt kulturpolitiskt sammanhang belyser uppsatsen hur en filmregional aktör arbetar med eventkultur som strategi för att stärka filmens synlighet, publikrelationer och position i Skåne. Fokus riktas mot olika former av filmevent och återkommande koncept – såsom M:Dox, CineSkåne och Filmbar – samt deras funktion i ett bredare ekosystem av filmfestivaler och publika satsningar.\n\nStudiens centrala syfte är att förstå vilka mål och logiker som präglar Film i Skånes eventarbete, samt vilka möjligheter och spänningar som uppstår när filmkultur i allt högre grad förmedlas genom tidsbegränsade, upplevelsebaserade format. Genom att närläsa verksamhetens praktiker synliggörs hur eventen kan fungera som mötesplatser för publik, bransch och institutioner, där nätverkande, identitetsskapande och regional profilering blir lika viktiga som själva filmvisningen. Samtidigt problematiseras eventifieringen som fenomen: när resurser, uppmärksamhet och kulturkonsumtion koncentreras till event riskerar kontinuitet, fördjupning och långsiktigt publikbyggande att hamna i skymundan.\n\nUppsatsen placerar Film i Skånes arbete i en kontext av regionalisering, där kulturella initiativ ofta motiveras av både kulturpolitiska och utvecklingspolitiska ambitioner. Sammantaget visar studien hur eventverksamheten kan förstås som ett uttryck för en samtida kulturtrend där upplevelser och synlighet blir centrala verktyg för att skapa legitimitet och attraktionskraft. \"Eventbubblan i syd\" framstår därmed både som en möjlighet att vitalisera filmkulturen i regionen och som en utmaning som väcker frågor om hållbarhet, prioriteringar och filmkulturens långsiktiga förankring.",
        "is_human": False,
    },
    {
        "title": "\"Men det här är ju kvinnogöra\": En studie om hur lärare ser på och praktiserar jämställdhet och genus i HKK",
        "text": "Kvinnor och män har historiskt sett ansetts ha olika fysiska och psykiska förutsättningar och den norm i samhället som vi levt efter har varit mannens. Kvinnans plats var i hemmet som maka; mor och husfru. I och med samhällets utveckling förändrades den könsspecifika arbetsfördelningen och kvinnan började förvärvsarbeta. Hem – och konsumentkunskap har gått från flickämne i skolan till obligatoriskt för alla där skolan ska främja jämställdhet och jämlikhet mellan könen. Denna rapports syfte är att undersöka hur yrkesverksamma lärare inom hem – och konsumentkunskap ser på jämställdhet och genus inom sitt ämne; vad som kan påverka hur de arbetar med jämställdhet mellan könen i sin undervisning samt hur de praktiskt genomför detta. Sju behöriga lärare från norra Sverige deltog i enskilda intervjuer. De hade olika åldrar och längd på yrkesverksamhet. Intervjuerna spelades in; transkriberades och analyserades med kvalitativ innehållsanalys. Respondenterna beskrev en pågående förändring av stereotypa könsmönster både i skolan och i hemmet men att traditionella könsroller fortfarande reproduceras. Hkk beskrevs som ett skolämne där lärandet till stor del sker i praktiskt arbete där roller synliggörs som kan ge förutsättningar till diskussion och reflektion om jämställdhet. Respondenterna var dock eniga om att jämställdhetsarbete i skolan ställer krav på lärarrollen och sig själv som verktyg; exempelvis vid bedömning av flickor och pojkar. Slutsatsen av denna studie är att lärare i hkk anser att stereotypa könsmönster har förändrats men fortfarande reproduceras traditionella könsroller i skolan. Yngre generationer frångår alltmer de traditionella könsrollerna då jämställdhet blir en allt mer naturlig del i samhället; dock kvarstår det arbete i jämställdhetsfrågan då det fortfarande finns skillnader mellan könen i betyg.",
        "is_human": True,
    },
]

THREADS_QUIZ = [
    {
        "question": "Snus marknadsförs i USA – kommer det funka den här gången?",
        "comments": [
            {
                "text": "Det är svårt att säga. Snus har sin charm och kan locka många, särskilt rökare som söker ett alternativ. Men den amerikanska marknaden är tuff och hälsodebatten kring tobaksprodukter är intensiv. Det hänger på hur bra de lyckas med både marknadsföring och att övervinna motståndet.",
                "is_human": False,
            },
            {
                "text": "Forsokte snusa lite Copenhagen och Camel snus nar jag bodde i Seattle, men det ar verkligen inte samma grej som hemma. Pasarna ar alldeles for stora och, som robeph skriver, sa ar materialet lite off ocksa. Tror for ovrigt inte att konceptet ar sa slaende nar det inte ar en del av kulturen som den ar i Sverige.",
                "is_human": True,
            },
            {
                "text": "Kommenterade lite med en snubbe om just camel snus, han var inte så förtjust, men varför inte? med alla sluta röka på offentliga platser lagar så.",
                "is_human": True,
            },
            {
                "text": "Det är svårt att säga. Snus har ju inte tagit fart på samma sätt som i Sverige, men med rätt marknadsföring och hälsobudskap kan det kanske bli en succé. Många amerikaner är ju också nyfikna på svenskt snus, så man får se hur det utvecklas!",
                "is_human": False,
            },
        ],
    },
    {
        "question": "Vad fick du i julklapp i år Sweddit?.",
        "comments": [
            {
                "text": "Jag fick några riktigt bra böcker som jag längtar efter att läsa, plus en go hoodie som är perfekt för kalla vinterkvällar. Hade även en mysig tid med nära och kära, mycket kvalitetstid! Hur var era julklappar?",
                "is_human": False,
            },
            {
                "text": "Haha nej du Jävelberg, mig lurar du men ingen annan!",
                "is_human": True,
            },
            {
                "text": "Jag fick en riktigt bra trådlös hörlur och ett nytt pussel! Hörlurarna är perfekt när jag vill lyssna på musik utan sladdar, och pusslet är just den svåra utmaningen jag behöver för att koppla av under vintermörkret.",
                "is_human": False,
            },
            {
                "text": "Ett lakan av syster. Ett badlakan av mor. En dagsedel av far.",
                "is_human": True,
            },
        ],
    },
    {
        "question": "Det slog mig att juldagen inte har samma \"självklara\" traditioner som julafton, visst firar många julafton olika men tex julklapparna öppnas ju alltid den 24:e. Vad är juldagen för dig?",
        "comments": [
            {
                "text": "Lång jävla promenad och dåliga TV-program.",
                "is_human": True,
            },
            {
                "text": "För mig är juldagen en dag av lugn och gemenskap. Efter all fest och stök på julafton, är det skönt att bara vara med familjen, äta lite rester och kanske gå ut på en promenad. Det är en dag att njuta av stillheten och reflektera över allt som har varit.",
                "is_human": False,
            },
            {
                "text": "Olika från år till år, beror på hos vem julafton firas. Det blir endel bök när både jag och sambon är skilsmässobarn. Men en promenad vill jag gärna ta om vädret tillåter.",
                "is_human": True,
            },
            {
                "text": "Om man är uppvuxen på någon liten ort och åker dit hem till föräldrar över jul så brukar juldagen vara utgång på \"stan\" och man träffar folk man växte upp med.",
                "is_human": True,
            },
        ],
    },
    {
        "question": "Vad är några roliga svenska uttryck som du använder dig av?",
        "comments": [
            {
                "text": "Nu är det kokta fläsket stekt.",
                "is_human": True,
            },
            {
                "text": "Blöt under fötterna.",
                "is_human": True,
            },
            {
                "text": "Ett av mina favoriter är \"Kasta inte pärlor för svin\" – det påminner mig om att uppskatta det som verkligen betyder något. Sen har vi \"Sätta kniven i ryggen\" för att beskriva svikande vänner, alltid en träffande bild! Svensk humor är verkligen unik.",
                "is_human": False,
            },
            {
                "text": "Ajsing bajsing!",
                "is_human": True,
            },
        ],
    },
    {
        "question": "Bokhandlar i Göteborg? Kan någon rekommendera en bra bokhandeln i Göteborg med läsvänlig atmosfär? Letar efter någonstans att spendera några timmar i helgen över en kopp kaffe, utan att vara bråttom att köpa (liksom Barnes & Noble, kanske) ...",
        "comments": [
            {
                "text": "Jag vet inte om nån sådan bokhandel. Stadsbiblioteket däremot har ett kafé som är helt ok. Kan vara värt att testa kanske?",
                "is_human": True,
            },
            {
                "text": "Hej! Jag rekommenderar att du besöker Bokskåpet på Chalmers eller Akademibokhandeln i Nordstan. Båda har en mysig atmosfär och en bra kaffehörna. Perfekt för att bläddra bland böcker och ta det lugnt en lördag!",
                "is_human": False,
            },
            {
                "text": "Lite kuriosa: Bokhandel i plural heter \"boklådor\". Låter knäppt, jag vet, men det är faktiskt sant.",
                "is_human": True,
            },
            {
                "text": "Absolut, kolla in Kronans bokhandel i Linné! De har en mysig atmosfär, bra utbud och ofta sittplatser där man kan bläddra i böcker. Dessutom ligger det nära mysiga kaféer. Perfekt för en avslappnad helg!",
                "is_human": False,
            },
        ],
    },
    {
        "question": "Varför tappade de rödgröna makten till att börja med? Jag är 94a så är rätt ung, men jag undrar hur det kommer sig att de rödgröna har regerat Sverige under en väldigt lång period och S alltid har varit det traditionellt största partiet, så kommer Alliansen från ingenstans och plötsligt vinner, och får dessutom sitta kvar i två mandatperioder. Finns det någon speciell anledning till att Alliansen vann? Typ någon dålig reform som sossarna gjorde eller något liknande.",
        "comments": [
            {
                "text": "Det handlar nog om en kombination av flera faktorer. Många väljare kände sig trötta på det rödgröna styret efter så många år, och Alliansen lyckades presentera sig som ett fräscht alternativ med fokus på skattesänkningar och jobbskapande. Dessutom påverkade ekonomin mycket, speciellt med krisen som kom strax innan valet.",
                "is_human": False,
            },
            {
                "text": "Det brukade vara mycket större skilnad mellan sossarna och moderaterna, men under sin tid vid makten så rörde sig sossarna allt längre höger ut. De var ju tex de som öpnnade upp för friskolor och på så sätt startade hela den här kvasimarknadstrenden. Samtidigt så rörde sig moderaterna in mot mitten. Allt prat om nya moderaterna och att de skulle vara ett arbetar parti. När de två sidorna blev allt mer lika, så blev det enklare för väljarna att byta sida, och det räcker med en liten sak som att Göran Persson ser trött ut för att det ska bli regeringsskifte.",
                "is_human": True,
            },
            {
                "text": "Göran Persson upplevdes som väldigt trött på sitt jobb inför valrörelsen 2006 vill jag minnas. Alliansen kom med idéerna. Det hade varit mycket prat om hur man kunde ha nästan samma inkomst på bidrag som på ett låginkomstjobb och därför kunde man sälja in jobbskatteavdraget.",
                "is_human": True,
            },
            {
                "text": "Det är en intressant fråga! Många menar att Alliansens seger delvis berodde på missnöje med S under finanskrisen, samt att de lyckades profilera sig som ett mer borgerligt alternativ. Dessutom spelade oppositionens enighet stor roll. Reformer som rör skolor och vård kritiserades också, vilket bidrog till att väljare sökte förändring.",
                "is_human": False,
            },
        ],
    },
    {
        "question": "Vad är egentligen Sverigedemokraterna?",
        "comments": [
            {
                "text": "Om man läser idéprogrammet så inser man rätt snabbt att det handlar om ett socialkonservativt parti - med vissa populistiska drag. Sen är det väl litet av ett myserium vad varje enskild medlem vill men så är det ju med socialdemokraterna också.",
                "is_human": True,
            },
            {
                "text": "Sverigedemokraterna är ett politiskt parti i Sverige som ofta kopplas till nationalism och invandringskritik. De har vuxit snabbt de senaste åren och väcker många känslor, både positiva och negativa. För vissa står de för opinionsskifte, medan andra ser dem som ett hot mot den svenska värdegrunden.",
                "is_human": False,
            },
            {
                "text": "Sverigedemokraterna är ett svenskt politiskt parti som grundades 1988 och har fokuserat på frågor kring migration och integration. Deras politik har varit kontroversiell och de har fått både stöd och kraftig kritik. Många ser dem som ett populistiskt parti, medan andra anser att de representerar en viktig röst för vissa medborgare.",
                "is_human": False,
            },
            {
                "text": "\"Det här är ett parti som vill avskaffa centrala delar av demokratin.\" Vilken demokrati? Sverige röstade höger, fick vänster. Stefan Löfven har satt sig över demokratin. Ja, man kommer förmodligen behöva gå hårt fram och sparka ut en del stofiler för att få till nödvändiga åtgärder. Men att sitta och nörda ner sig i vad som kan hända om SD får ensam majoritet är faktiskt duktigt efterblivet.",
                "is_human": True,
            },
        ],
    },
    {
        "question": "Vilka enskilda politiker ger bäst intryck?",
        "comments": [
            {
                "text": "Typ ingen. Svenska åsikts- och beteendekorridoren är så trång att ingen riksdagspolitiker ens gör ett mänskligt intryck. Som vänstersnubben sa, Hanif Bali är lite (trollaktigt) sympatiskt, skulle kanske hänga med honom nån gång.",
                "is_human": True,
            },
            {
                "text": "Jag skulle nog säga att Jimmie Åkesson är bäst och efter det kommer Annie Lööf följt av Jonas Sjöstedt.",
                "is_human": True,
            },
            {
                "text": "Gillar verkligen Mattias Karlsson. Känns stabil och seriös. Och verkar som en riktig familjefar utanför politiken också.",
                "is_human": True,
            },
            {
                "text": "Det beror verkligen på vad man värderar mest. Jag tycker att politiker som är genuina och lyssnar på medborgarna, som exempelvis Annie Lööf och Stefan Löfven, ger bra intryck. Det handlar också om att vara transparent och stå för sina åsikter. Vad tycker ni?",
                "is_human": False,
            },
        ],
    },
    {
        "question": "Lumpen – har ni gjort den? Jag tycker att vi måste blåsa lite liv i denna reddit, så jag föreslår att vi börjar snacka om det. Gjorde själv inte militärtjänst, var upptagen med andra dumma saker vid den åldern. Dock ångrar jag det väldigt mycket, tror att den hade varit en upplevelse. Åsikter/erfarenheter?",
        "comments": [
            {
                "text": "Jag gjorde lumpen för några år sedan och det var verkligen en blandning av hårt arbete och gemenskap. Självklart finns det saker som var jobbiga, men jag ångrar inte en sekund. Det var en chans att träffa nya människor och lära sig mycket om sig själv. Rekommenderar verkligen!",
                "is_human": False,
            },
            {
                "text": "Jag gjorde inte lumpen, och jag ångrar det inte. Inledningsvis blev jag placerad som civilpliktig i Porjus vattenkraftverk, och det hade säkert varit intressant med tanke på min senare utbildning och mitt nuvarande jobb, men det hade varit 11 långa månader i en liten by i Norrland. Min tjänst drogs in, och jag valde att plugga, resa och jobba det året istället, och känner att den tiden gav mer.",
                "is_human": True,
            },
            {
                "text": "Gjorde lumpen för några år sen och det var både skitjobbigt och bland det roligaste jag gjort. Mycket väntan, lite \"pang pang\", men framför allt rutin, kyla, sömnbrist och att lära sig funka i grupp även när man är helt slut. Man får en speciell gemenskap som är svår att hitta annars, plus att man växer som person (klyscha men sant). Samtidigt är det inte för alla, vissa hatade varje minut. Om du ångrar dig: kolla Hemvärnet eller frivilligorganisationer, eller sök GU nu om du är sugen. Det är inte \"för sent\" bara för att man missade 18-årsåldern.",
                "is_human": False,
            },
            {
                "text": "Känns ibland som jag är en av de sista som hunnit med tre repövningar utöver originallumpen vid 42 års ålder. Fick ett sorgkantat brev från ÖB för ett par år sedan där han beklagade att det inte fanns någon plats för mig längre. Måste erkänna att jag hade görskoj 80% av tjänstgöringen faktiskt.",
                "is_human": True,
            },
        ],
    },
    {
        "question": "Vad hände med Reddit Meetup Day? Det var ju under vår nationaldag har jag för mig. Tycker att det borde styras upp ifall det inte blev något :D",
        "comments": [
            {
                "text": "Ja, det känns som att det har blivit lite tyst om det! Skulle vara kul om de kunde fixa till det igen. Kanske vi borde börja peppa för nästa år?",
                "is_human": False,
            },
            {
                "text": "Ja, Reddit Meetup Day kändes som en kul grej, speciellt på vår nationaldag! Det är synd att det inte blev av. Förhoppningsvis kan någon ta initiativ och organisera något för att samla alla Redditanvändare nästa år. Hade varit grymt att träffas och diskutera allt möjligt!",
                "is_human": False,
            },
            {
                "text": "Ja, Reddit Meetup Day har tyvärr blivit lite av en bortglömd tradition. Det vore verkligen kul om folk tog initiativ att planera det igen! Skulle vara trevligt att träffa likasinnade och ha en kul dag tillsammans. Hoppas på bättre organisering framöver!",
                "is_human": False,
            },
            {
                "text": "Ja, det var en rolig tradition! Synd att den verkar ha försvunnit. Det skulle vara kul om någon tog initiativ till att ordna en ny meetup, speciellt på nationaldagen. Kanske vi kan skapa något lokalt igen? Hoppas fler är på!",
                "is_human": False,
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# DETECTION FUNCTIONS
# ---------------------------------------------------------------------------

def _call_deepseek(prompt: str) -> str:
    """Call DeepSeek chat API with exponential-backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            # send prompt to API
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            # return text from API response
            return response.choices[0].message.content
        except Exception as e:
            # retry mechanics - if hit max retries, raise error. else, wait and retry
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f"  API error ({e}), retry in {wait}s...", flush=True)
            time.sleep(wait)
    return ""


def detect_formal(abstract_text: str) -> str:
    """Ask DeepSeek to classify a bachelor's thesis abstract as AI or human."""
    prompt = (
        "Du deltar i en studie som handlar om att identifiera AI-genererad akademisk text på svenska.\n\n"
        "Analysera följande kandidatuppsatsabstrakt och avgör om det är skrivet av en AI "
        "eller av en människa. Titta på saker som språklig variation, formuleringsval, "
        "meningsbyggnad, och om texten känns autentiskt akademisk eller mallartad.\n\n"
        f"Abstrakt:\n{abstract_text}\n\n"
        "Svara med JSON i exakt detta format (ingen annan text utanför JSON-blocket):\n"
        '{"classification": "AI" or "MÄNNISKA", "confidence": <0-100>, '
        '"reasoning": "<kort förklaring på 50-100 ord>"}'
    )
    return _call_deepseek(prompt)


def detect_informal(comment_text: str) -> str:
    """Ask DeepSeek to classify a Reddit comment as AI or human."""
    prompt = (
        "Du deltar i en studie som handlar om att identifiera AI-genererad akademisk text på svenska.\n\n"
        "Analysera följande Reddit-kommentar och avgör om den är skriven av en AI "
        "eller av en människa. Titta på saker som informellt/vardagligt språk, stavfel, "
        "personliga referenser, naturlig röst, och om texten känns genuin eller generisk.\n\n"
        f"Kommentar:\n{comment_text}\n\n"
        "Svara med JSON i exakt detta format (ingen annan text utanför JSON-blocket):\n"
        '{"classification": "AI" or "MÄNNISKA", "confidence": <0-100>, '
        '"reasoning": "<kort förklaring på 50-100 ord>"}'
    )
    return _call_deepseek(prompt)


def parse_response(raw: str) -> tuple[str, int, str]:
    """
    Extract classification, confidence and reasoning from DeepSeek's JSON response.
    Returns ("AI"|"MÄNNISKA"|"UNKNOWN", confidence, reasoning).
    """
    text = raw.strip()
    # Strip markdown code fences if present
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
    try:
        data = json.loads(text)
        classification = data.get("classification", "UNKNOWN").upper()
        if "AI" in classification and "MÄNNISKA" not in classification:
            classification = "AI"
        elif "MÄNNISKA" in classification or "MANNISKA" in classification:
            classification = "MÄNNISKA"
        else:
            classification = "UNKNOWN"
        return classification, int(data.get("confidence", 50)), data.get("reasoning", "")
    except json.JSONDecodeError:
        upper = text.upper()
        if "MÄNNISKA" in upper or "MANNISKA" in upper or "HUMAN" in upper:
            return "MÄNNISKA", 50, text[:200]
        if "AI" in upper:
            return "AI", 50, text[:200]
        return "UNKNOWN", 0, text[:200]


# ---------------------------------------------------------------------------
# ACCURACY MEASUREMENT
# ---------------------------------------------------------------------------

def measure_accuracy(results: list[dict]) -> dict:
    """
    Compute overall, per-type, and AI-detection metrics.

    Each result dict must contain:
        type       – "abstract" or "comment"
        is_human   – bool  (ground truth)
        predicted  – "AI" | "MÄNNISKA" | "UNKNOWN"
        correct    – bool
    """
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r["correct"])

    abstract_results = [r for r in results if r["type"] == "abstract"]
    comment_results  = [r for r in results if r["type"] == "comment"]

    abstract_correct = sum(1 for r in abstract_results if r["correct"])
    comment_correct  = sum(1 for r in comment_results  if r["correct"])

    # Treating AI-detection as the positive class
    tp = sum(1 for r in results if not r["is_human"] and r["predicted"] == "AI")
    fp = sum(1 for r in results if     r["is_human"] and r["predicted"] == "AI")
    tn = sum(1 for r in results if     r["is_human"] and r["predicted"] == "MÄNNISKA")
    fn = sum(1 for r in results if not r["is_human"] and r["predicted"] == "MÄNNISKA")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "total_accuracy":    correct / total,
        "abstract_accuracy": abstract_correct / len(abstract_results) if abstract_results else 0.0,
        "comment_accuracy":  comment_correct  / len(comment_results)  if comment_results  else 0.0,
        "precision_ai": precision,
        "recall_ai":    recall,
        "f1_ai":        f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "correct": correct,
        "total":   total,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []

    # ---- First half: abstracts (formal) ----
    print("=" * 60)
    print("FORMAL DETECTION: BACHELOR'S THESIS ABSTRACTS")
    print("=" * 60)
    for i, item in enumerate(ABSTRACTS_QUIZ):
        label = "Human" if item["is_human"] else "AI"
        print(f"[{i+1:02d}/{len(ABSTRACTS_QUIZ)}] {item['title'][:55]}...", flush=True)
        raw = detect_formal(item["text"])
        classification, confidence, reasoning = parse_response(raw)
        predicted_human = (classification == "MÄNNISKA")
        correct = (predicted_human == item["is_human"])
        tick = "✓" if correct else "✗"
        print(f"       Ground truth: {label:5s}  |  Predicted: {classification}  "
              f"(conf: {confidence})  {tick}")
        results.append({
            "type":        "abstract",
            "title":       item["title"],
            "text":        item["text"][:80] + "...",
            "is_human":    item["is_human"],
            "predicted":   classification,
            "confidence":  confidence,
            "reasoning":   reasoning,
            "correct":     correct,
        })

    # ---- Second half: Reddit comments (informal) ----
    print()
    print("=" * 60)
    print("INFORMAL DETECTION: REDDIT COMMENTS")
    print("=" * 60)
    for t_idx, thread in enumerate(THREADS_QUIZ):
        print(f"\nThread {t_idx+1}: {thread['question'][:70]}...")
        for c_idx, comment in enumerate(thread["comments"]):
            label = "Human" if comment["is_human"] else "AI"
            print(f"  Comment {c_idx+1}...", end=" ", flush=True)
            raw = detect_informal(comment["text"])
            classification, confidence, reasoning = parse_response(raw)
            predicted_human = (classification == "MÄNNISKA")
            correct = (predicted_human == comment["is_human"])
            tick = "✓" if correct else "✗"
            print(f"Ground truth: {label:5s}  |  Predicted: {classification}  "
                  f"(conf: {confidence})  {tick}")
            results.append({
                "type":        "comment",
                "title":       f"Thread {t_idx+1}, Comment {c_idx+1}",
                "text":        comment["text"][:80] + "...",
                "is_human":    comment["is_human"],
                "predicted":   classification,
                "confidence":  confidence,
                "reasoning":   reasoning,
                "correct":     correct,
            })

    # ---- Accuracy report ----
    print()
    print("=" * 60)
    print("ACCURACY REPORT")
    print("=" * 60)
    m = measure_accuracy(results)
    print(f"Overall accuracy  : {m['total_accuracy']:.1%}  ({m['correct']}/{m['total']})")
    print(f"Abstract accuracy : {m['abstract_accuracy']:.1%}")
    print(f"Comment accuracy  : {m['comment_accuracy']:.1%}")
    print(f"Precision (AI+)   : {m['precision_ai']:.1%}")
    print(f"Recall    (AI+)   : {m['recall_ai']:.1%}")
    print(f"F1 score  (AI+)   : {m['f1_ai']:.3f}")
    print(f"TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")

    # ---- Save results ----
    out_path = os.path.join(
        _root, "src", "3_ai_detection_scripts", "detection_results_deepseek.csv"
    )
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nResults saved → {out_path}")
