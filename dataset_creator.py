import json


fikralar = [
    {
        "prompt": "Bilim adamı pirelerle deney yapıyor.",
        "story": "Bir bilim adamı pirelerle deney yapıyor. Pireye \"Sıçra\" diyor. Pire sıçrıyor Pirenin kanatlarını koparıyor ve \"Sıçra\" diyor, pire yine sıçrıyor!.. Raporunu şöyle yazıyor; \"Pire kanatlarını kopmuş olarak zıpladı.\"\nAyaklarını koparıyor ve \"Sıçra\" diyor, pire zıplayamayınca 2. raporu şöyle yazıyor; \"Pirenin ayakları kopunca kulakları duymuyor.\""
    },
    {
        "prompt": "Temel ilk kez helikopter görür.",
        "story": "İlk defa helikopter gören Temel, Dursun'a sormuş;\n- Dursun, bu nedur da? Dursun, gayet sakin cevap vermiş;\n- Haçan, bu olsa olsa bin yaşında bir sinek dur."
    },
    {
        "prompt": "Akıl hastası taburcu edilirken doktora ayakkabı yapmayı teklif eder.",
        "story": "Doktor, akıl hastasının iyileştiğine karar verip akıl hastasını taburcu etmiş. Buna çok sevinen hasta gelip doktoruna şöyle demiş;\n-\"Doktor Bey, size hediye ayakkabı yapacağım ama sorayım dedim, topuğu önde mi olsun arkada mı?"
    },
    {
        "prompt": "Çocuk soprano konserinde şefi yanlış anlar.",
        "story": "Ünlü bir sopranonun konserine giden baba oğul ilgiyle konseri dinliyorlardı. Bir ara çocuk merakla babasına sordu:\n\"Baba, öndeki amca elindeki sopayla niye kadını korkutuyor ?\" Baba;\n\"Korkutmuyor oğlum, yönetiyor!\"\n\"Eee, peki o zaman kadın niye avaz avaz bağırıyor \""
    },
    {
        "prompt": "Küçük Ahmet bakkaldan küçük yumurta alır, eksik para verir.",
        "story": "Küçük Ahmet, bakkala öfkeyle sordu:\n- Neden hep küçük yumurta veriyorsun?\n- Taşıması , kolay olur da ondan.\nAhmet eksik para verip yumurtaları alıp giderken bakkal seslendi:\n- Ama sen eksik para verdin.\nKüçük çocuk arkasına dönüp gülerek: \" Para daha çabuk sayılır da...\""
    },
    {
        "prompt": "Nuri okuldan mor gözle döner, annesine açıklama yapar.",
        "story": "Nuri okuldan eve bir gözü mosmor dönmüştü. Annesi çıkıştı:\n\"Aşk olsun yine mi dövüştün okulda? \"Şey büyük bir çocuğun küçüğü dövmesine engel olmaya çalıştım da anneciğim.\"\n\"Aferin bak bu cesaret işi. Kimdi o küçük?\"\nNuri gayet sakin;\n\" Ben! \""
    },
    {
        "prompt": "Adamın evi yanar, komşusu haber verir.",
        "story": "Vurdumduymaz bir adamın evi yanmış. Komşusu koşarak yanına gelmiş.\n\" Koş efendi, evin yanıyor. \"\nAdam sakince cevap vermiş :\n\"Ev işlerine karım bakıyor.\""
    },
    {
        "prompt": "Küçük çocuk karşısındakine süre verir, sonra pazarlık yapar.",
        "story": "Küçük çocuk, kendinden daha büyük olana yan yan bakarak: \"Söylediğin sözü geri alman için sana beş dakika süre veriyorum!\" dedi. Öbürü kabararak:\n\"Bak hele sen. Peki beş dakika sonra sözümü geri almazsam ne olacak?\" diye diklendi. Küçük çocuk biraz düşündükten sonra:\n\"Peki söyle ne kadar zaman istiyorsun?\" dedi."
    },
    {
        "prompt": "Sergi açılışında romancı ve ressam sohbet eder.",
        "story": "Bir sergide ünlü romancı, ressam arkadaşına:\n\"Kutlarım sergi açılışına bakanlar gelmiş\"\nBunun üzerine Ressam:\n\"Ne önemi var ki, bakanlar geleceğine, keşke biraz da alanlar gelseydi. \" der."
    },
    {
        "prompt": "Doktor hastaya kötü haber verir, sonra daha kötüsünü söyler.",
        "story": "Doktor, hastasına o güne dek yaptığı tahlillerin sonuçlarını açıklayacak;\n\"Size bir kötü, bir de daha kötü haberim var. Önce kötü haberi vereyim. Test sonuçlarına göre 24 saatlik ömrünüz kalmış.\" deyince adam yıkılır,\n\"Hayır, olamaz. Buna inanamıyorum: Fakat bundan daha kötü haber nasıl olabilir? \"deyince hasta, doktorun yanıtı kısa olur;\n\"Dünden beri size ulaşmaya çalışıyorum.\""
    },
    {
        "prompt": "Dil bilgisi dersinde öğretmen 'Bağırmadım, bağırmadın, bağırmadı' örneğini sorar.",
        "story": "Dil bilgisi dersinde öğretmen öğrencilere sordu:\n-\"Bağırmadım, bağırmadın, bağırmadı\" deyince ne anlarsınız? diye sordu.\nKimseden çıt çıkmıyordu. Öğretmen bütün öğrencilerin birden parmak kaldırmasını beklediği için, hayal kırıklığına uğradı.\nNeden sonra ön sıralardan Temel ayağa kalkarak söz hakkı istedi. Öğretmen söz verince de cevapladı:\n-Önemli bir durum yok efendim. Hiç kimse bağırmamıştır."
    },
    {
        "prompt": "Temel İstanbul’a yeni taşındığında kapıcıyla karşılaşır.",
        "story": "Temel İstanbul'a yeni taşınmış. Kapıcı sabah kapıyı çalmış.\nTemel, kimseyi beklemediğinden merakla kapıya yönelmiş ve seslenmiş;\n-Kim o?\nKapıcı:\n-Çöp! diye bağırmış...\nTemel gayet sakin ve kibar bir dille konuşmuş:\n-İhtiyacımız yok..."
    },
    {
        "prompt": "Boksör rakibine yumruk atamaz, menejeri öneride bulunur.",
        "story": "Boks maçı hayli heyecanlı geçiyordu. İki boksör ringde kıyasıya dövüşüyorlardı. Ama birinin durumu pek kötüydü. Yumrukları havayı dövüyor, bir teki bile rakibine değmiyordu. Raund arasında menejerine sordu:\n\"Maçı almam için bir şansım var mı?\"\nMenejeri bir yandan terini kurularken:\n\"Elbette var,\" diye cevap verdi. \"Etrafındaki havayı dövmeye devam et. Böylelikle rakibini zatüreden öldürebilirsin.\""
    },
    {
        "prompt": "Temel kaç rulo duvar kâğıdı alacağını sorar.",
        "story": "- Temel bey, dairelerimiz aynı genişliktedir. Sen evi duvar kâğıdıyla kaplattın? Ben de evi dekore edeceğim de. Ne kadar duvar kağıdı aldın?\n- On yedi top aldum.\nKomşu da duvar kâğıdını alır, evi kaplatır, ama epeyce de kâğıt elinde kalır.\n- Yahu Temel, ben de on yedi top aldım ama, yedi top arttı!\n- Eyi, benum da o kadar artmıştı!"
    },
    {
        "prompt": "Fadime gece dörde kadar uyumaz, sebebi Temel'dir.",
        "story": "- Yahu Recep, bizum Fadume'nun çok köti bi huyi vardur. Gece dörde kadar uyumayı!\n- Temelcuğum, peki o saate kadar ne yapayi?\n- Penum eve gelmemi bekliyor!"
    },
    {
        "prompt": "Yabancı, tavuk ezdiği köyde sahibini bulmaya çalışır.",
        "story": "Karadeniz'de bir köyden geçen bir yabancı arabasıyla bir tavuk ezer. Kaçacaktır ama korkar. Dönüşte gene aynı köyden geçecektir. En iyisi sahibini bulup parasını vermek. Muhtarı bulur durumu anlatır. Tavuğu verir. Ancak tavuk dümdüz olmuştur. Muhtar köylüleri tek tek çağırır. Tavuğu gösterir. Hiç kimse tavuğa sahip çıkmaz. Muhtar sonucu yabancıya açıklar:\n- Bizim köyde yamyassı tavuk yoktur."
    },
    {
        "prompt": "Temel fırından ekmek almak ister.",
        "story": "Temel Karadenizlinin fırınından bir ekmek alacak. Kafasını fırından içeri uzatır:\n- Ha oradan bi ekmek vermeni rica edeyirum!\n- Ula parasını verecek misun?\n- Elbette vereceğum.\n- Haçan parasını vereceksen ne diye rica edeyisun?"
    },
    {
        "prompt": "Temel, Erdal İnönü'yü mitingde görür.",
        "story": "İsmet Paşanın oğlu Erdal İnönü, bir seçim mitingi için Rize'ye gider. Kürsüde konuşan ince zayıf uzun boylu İnönü'yü gören Temel sorar:\n- Habu konuşan adam da kimdur?\nDerler ki: İsmet İnönü'nün oğlu Erdal'dır!\n- Uy desene Paşanun çok günahını almışuz. Rahmetli II. Dünya Savaşı yıllarında bizleri çok aç bırakmıştı. Baksanıza ne kadar adaletli davranmuş, kendi uşağını da aç bırakarak ne hale getirmiş!"
    },
    {
        "prompt": "Temel, Fadime'nin ameliyat faturasını kayınbabasına gönderir.",
        "story": "Temel, karısı Fadime'yi bademcik ameliyatı yaptırmıştı. Hastaneden taburcu edilirken, doktor Temel'e bazı tavsiyelerde bulunur ve son olarak der ki;\n- Aslında bu ameliyat gecikmiş, daha çocukken yapılmalıydı.\nTemel hemen söze girer:\n- O zaman faturayı kayınbabamı gönder de, hasabını o ödesun!"
    },
    {
        "prompt": "Cemaat içinde Temel yanlış anlar.",
        "story": "Hoca, minberden cemaate hitaba başlar:\n- Ey cemaat-i müslimin, deyince: Arkalardan Temel, cevap verir:\n- Efendum! Bağa mi deyisun?"
    },
    {
      "prompt": "Temel kendini ağaca belinden asar, Dursun müdahale eder.",
      "story": "Dursun evinden çıktığında birde bakar ki komşusu Temel kendini belinden ağaca asmış halde duruyor. Hemen gidip ipi ağaçtan çözer. Komşusunu ağaçtan indirdikten sonra merakla sorar:\n-Ha sen ne yapayudun öyle?\n-Hiç kendimi asaydum...\n-Ha uşağum, penum pildiğum insan poynundan asılayi.\nTemel üzgün ve çaresiz bir halde komşusu Dursun'a baktıktan sonra cevap verir:\n-Ben de öyle yapmişudum. Ama ipu poynima pağladığum zaman bi türlü nefes alamayrum."
  },
  {
      "prompt": "Temel kırtasiyeden roman ister, arabanın dışarıda olduğunu söyler.",
      "story": "Temel kırtasiye'ye girmiş, tezgahtara:\n-Pana pir roman lazum, demiş.\nKırtasiye tezgahtarı sormuş:\n-Efendim agır mı olsun hafif mi?\nTemel:\n-Farketmez, nasul olsa arabam dısarudadur."
  },
  {
      "prompt": "Temel FBI ajan seçmelerine katılır.",
      "story": "FBI gizli ajan eksikliğini giderebilmek için ajan seçmeleri yapmaya karar vermiş. Ve her gün üçer kişi çağırıp aralarından birini ajan olarak himayelerine alıyorlarmış. Seçimlerin 3. günü Temel de katılmış. Yanında bir İngiliz ve bir Amerikan varmış. Bunlardan ilk olarak kamuflaj olmalarını istemişler. İçinde sadece bir çuvalın bulunduğu boş bir odaya sokmuşlar ve burada gizlenmelerini söylemişler.\nİlk önce İngiliz girmiş. 5 dk. sonra odaya giren bir yetkili gitmiş içinde İngilizin saklandığı çuvala tekme atmaya başlamış. Hemen çuvalın içinden bir ses gelmiş: \"Miyaw, miyaw.\"\nİngilize ilk testi başarıyla geçtiğini söyleyip Amerikalıyı odaya koymuşlar. Amerikalı da aynı çuvala saklanmış. Biraz sonra yine odaya giren yetkili gitmiş ve çuvala bir tekme atmış. Çuvalın içinden: \"Hav, hav.\" diye bir ses gelmiş. Amerikalıyı da tebrik edip Temel'i odaya koymuşlar. 5 dk. sonra odaya giren aynı görevli gitmiş çuvala bir tekme atmış. Ama bir daha bir daha derken en sonunda çuvaldan cılız bir ses yükselmiş: \"Patateeeeesss\""
  },
  {
      "prompt": "Komutan, Temel'e her yönden gelen düşmana karşı plan sorar.",
      "story": "Manevra varmış. Temel elde tüfek yerde yatıyormuş. Komutan gelip sormuş:\n-Düşman önden gelirse ne yaparsın Temel?\nTemel cevaplamış. Şu yandan, bu yandan, arkadan gelirse, diye tekrar sormuş komutan. Temel bunları da cevaplamış.\nKomutan en sonunda:\n-Ya düşman tepeden gelirse? deyince,\nTemel dayanamamış ve:\n-Habu memleketin tek askeru ben miyum komitanum daa!"
  },
  {
      "prompt": "Karadenizli yalnızken kendi kendine konuşur mu?",
      "story": "Arkadaşı Karadenizliye sormuş:\n-Yalnızken kendi kendine konuşma huyun var mıdır?\n-Ben kendi kendime konumam, demiş Karadenizli. Adamı gözümün önüne getiririm, öyle konuşurum."
  },
  {
      "prompt": "Temel ormanda doğayı gösterir ama Dursun ağaçları görür.",
      "story": "Temelle Dursun ormanda yürüyorlar. Bir ara Temel Dursun’a sesleniyor:\n-Dursun ormanın güzelliğine bak.\nDursun:\n-Ağaçlardan göremiyorum ki."
  },
  {
      "prompt": "Temel Cemal'in ölen karısını sorar.",
      "story": "Temel uzun zamandır görmediği arkadaşı Cemal'le İstanbul'da karşılaşınca:\n-Usak nasılsun pakayum?\n-İyiyum...\n-Çocukların nasıldur?\n-Onlar da iyidur.\n-Peki karin nasıldur?\nTemel böyle sorunca Cemal'in birden yüzü değişir... Temel arkadaşının karısının geçen yıl öldüğünü hatırlayıp hemen şöyle der:\n-Yani hala ayni mezarda mi yatiyii?"
  },
  {
      "prompt": "Temel bulaşıkçı ilanı görüp başvurur.",
      "story": "Temel bir lokantanın önünden geçerken \"Bulaşıkçı Aranıyor\" ilanını görmüş. Hemen içeri girip patrona:\n-Pen ha purada pulasikçiluk yapapilirum.\nDemiş. Patron sormuş:\n-Kaç dil biliyorsun?\nTemel hiç duraksamadan cevap vermiş:\n-On tört\nÖnce biraz şaşıran patron sonra sinirlenmiş ve:\n-Sen benimle alay mı ediyorsun?\nTemel:\n-Valla önce sen paslattun..."
  },
  {
      "prompt": "Kamyon yüksekliğini gören Temel, etrafa bakar.",
      "story": "Tır şoförü Dursun ile muavin Temel kamyonlarına 6 metre yüksekliğinde havaleli mal yüklemiş gidiyorlarmış. Birden bir tünel ve önünde bir uyarı işareti:\n\"DİKKAT!! Azami Yükseklik 4 metre\"\nMuavin Temel, etrafa dikkatle bakmış. Sonra Dursun'a dönerek:\n-Bas gaza usta! Etrafta polis molis körinmeyu..."
  },
  {
      "prompt": "Cemal tabanca almak ister, beş kişilik der.",
      "story": "Cemal silahçı dükkanına girer:\n-Ha pi tapanca almak isteyrum.\nSatıcı sorar:\n-Nasıl bir tabanca?\nCemal:\n-Pes kisiluk..."
  },
  {
        "prompt": "Temel'in kol saati durur ve içinden bir karınca çıkar.",
        "story": "Temel'in kol saati durmuş. İçini açmış ve içinden ölü bir karınca çıkmış.\nTemel:\n-Uyy... Zaten pen tahmin etmiştum makinistun öltüğünü..."
    },
    {
        "prompt": "Hakim Temel'e neden sandalye ile vurduğunu sorar.",
        "story": "Hakim Temel'e sorar:\n-Niye adamın başına sandalyeyle vurdun?\nBoynu bükük Temel:\n-Ne yapayum, çaresizluk efendum. Masayı kaltıramatum ki..."
    },
    {
        "prompt": "Otobüste ayağına basılan Temel, adamın nereli olduğunu sorar.",
        "story": "Çok kalabalık bir belediye otobüsünde yolculuk eden Temel'in ayağına iri yarı bir adam basar....\nNasırı acıyan Temel, adamın yanına yaklaşır ve sorar:\n- Ula uşak, sen nerelisun?\nAdam, Temel'e bakar, nereli olduğunu söyler ve sonra da sorar:\n- Niye sordun?\n- Hiç, der Temel, bu cins ayular hangi memlekette yetişür diye merak ettum da..."
    },
    {
        "prompt": "Temel devekuşu avına çıkar ama devekuşlarını göremez.",
        "story": "Temel Avustralya'ya devekuşu avlamaya seyahate çıkıyor. Orada malzemelerini hazırlayıp maceraya atılıyor.\nBir virajı dönünce bakıyor 10-15 tane devekuşu.\nHemen arabayı durduruyor, silahını doğrultuyor. Devekuşları silahı görünce ürkerek kafalarını kuma gömüyorlar. Yani kendi akıllarınca saklanıyorlar.\nTemel etrafa bakıyor ve kendi kendine sinirli sinirli soruyor:\n-Ulan nereye gitti bu hayvanlar?..."
    },
    {
        "prompt": "Karadenizlilere neden perşembe günü fıkra anlatılmaz?",
        "story": "Soru: Karadenizlilere niçin perşembe günü fıkra anlatılmaz?\nCevap: Cuma namazında gülmesinler diye..."
    },
    {
        "prompt": "Temel'e 'internet nedir?' sorusu sorulur.",
        "story": "Temel iş için başvurmuş.\n- Önce bilgi testinden geçmen gerek, demişler ve sormuşlar,\n- İnternet ne demektir?\nTemel:\n- İşe ciremedum temektur."
    },
    {
        "prompt": "Temel trenin ters istikamete gittiğini öğrenir.",
        "story": "Temel trene binmiş, kontrol gelmiş, biletinin İstanbul'a olduğunu, trenin Ankara'ya gittiğini söylemiş.\nTemel kendinden emin:\n- Peçi maşinist yanlış istikamete cittigini piliy mi, demiş."
    },
    {
        "prompt": "Temel askerlikte geç kalan 39 kişiyi geçemez.",
        "story": "Temel askerlik yapıyormuş. Bölükte kırk ere izin vermişler.\nGeç kalırlarsa çadır hapsi var, ancak iyi bir mazeretleri olursa affedilecekler.\nKırk kişiden otuz dokuzu da geç kalmış, hep aynı mazeret:\n- Atla istasyona celeydum. At çatladı, tren kaçtı, geç kaldum.\nDerken kırkıncı da tamamlanmış, Temel çıkagelmiş.\n- Senin de mi atın çatladı, diye sormuşlar.\n- Hayır, demiş. Yoldaki otuz tokuz at leşini geçemedum."
    },
    {
        "prompt": "Temel İngilizce dersi alır, 'come' kelimesini anlamaz.",
        "story": "Temel İngiltere'ye gidecekmiş. Lisan öğrenmesi gerek.\nDershaneye yazılmış. İlk derste \"come\", yani \"gel\" demeyi öğretiyorlarmış.\nTemel bu işe akıl erdirememiş. Öğretmene demiş ki:\n- Bu nasıl iştir, come yazaysun, kam okuysun, peçi cel olduğunu nasıl anlaysun?"
    },
    {
        "prompt": "Zenci ile aynı otelde kalan Temel, yüzünü siyaha boyanmış bulur.",
        "story": "Temel aynı otelde kaldığı zenciyle \"Arap, arap!\" diye dalga geçermiş.\nBir akşam yatarken sabah beşte kaldırılmasını istemiş.\nZenci de gece usulca Temel'in odasına girip yüzünü siyaha boyamış.\nTemel sabah kalkıp aynaya bakınca:\n- Tuh peni kaldıracaklarına, yanlışlıkla Arabı kaldırmışlar, demiş."
    },
    {
        "prompt": "Temel, oğlunun defterinde neden yazı olmadığını sorar.",
        "story": "Temel her gün okula giden ve çalışan oğlunun defterinde tek bir yazı göremeyince nedenini sormuş.\nOğlan:\n- Öğretmen tahtaya ne yazarsa aynen teftere geçireyrum. O tahtayı silince pen de tefterimi sileyrum."
    },
    {
        "prompt": "Temel denizde batan gemiden sağ kurtulur.",
        "story": "Kaptan Temel'in gemisi çok kötü bir fırtınaya tutuluyor.\nBatan gemideki tayfalar ölüyor. Bir Temel sağ kalıyor:\n- Pen de ölseydim, puyuk facia olacaktı."
    },
    {
        "prompt": "Bakan denizin üstünde yürür ama gazetelerde yüzme bilmiyor yazar.",
        "story": "Bir ülkede bir bakan, kendisini gazetecilere hiç sevdirememişti. Ne yapsa makbule geçmiyor, basın her gün kendisiyle uğraşıyordu. Nihayet:\n-Öyle bir şey yapayım ki, gazeteciler mat olsun, diye düşündü ve ilan etti:\n-Pazar günü saat 10'da bakan denizin üzerinden yürüyerek geçeceğim.\nPazar sabahı saat 10'da tüm basın mensupları toplandılar orada. Bakan geldi ve elinde bastonuyla denizin üzerinde yürümeye başladı. Karşı kıyıya kadar da yürüdü geçti. Herkesin gözleri dehşetle açılmıştı.\nFakat ertesi günü tüm gazetelerde şu başlık okundu:\n-Bakan yüzme bilmiyor!"
    },
    {
        "prompt": "Ayakları kokan adam tiyatroya giderken çoraplarını değiştirme sözü verir.",
        "story": "Ayakları çok fena kokardı. Bir gün bir arkadaşına birlikte tiyatroya gitmelerini teklif etti.\n-Hay hay, dedi arkadaşı. Ama eve git, ayaklarını yıka ve temiz bir çorap giy. Söz mü?\nTiyatroya gittiler. Yerlerine oturdular.\nAradan beş on dakika geçmeden etrafındakiler mendillerini burunlarına götürmeye başladı.\n-Hani söz vermiştin, dedi arkadaşı.\n-Vallahi değiştirdim, dedi. İnanmazsın diye kirlileri de cebime koydum. Bak!..."
    },
    {
        "prompt": "Bir arkadaş bebeğin fil sütüyle beslendiğini söyler.",
        "story": "İki arkadaş teneffüste konuşuyorlardı:\n-Bugün bir bebek gördüm, fil sütü içerek bir haftada on yedi kilo almış.\n-Allah Allah, çok tuhaf. Kimin bebeğiymiş bu?\n-Filin!..."
    },
    {
        "prompt": "Tüm izciler aynı yaşlı kadını karşıdan karşıya geçirir.",
        "story": "Oymakbeyi, izci adaylarını karşısına toplamış, onlara izciliğin ilkelerini anlatmaya çalışıyordu:\n-Bakın çocuklar, dedi. Bir izci, her gün, hiç olmazsa bir kez birine yardımcı olmalıdır. Hastalara... Yaşlılara... Muhtaçlara...\nHer sabah okula geldiğiniz zaman size bir gün önce nasıl bir iyilik yaptığınızı soracağım. Tamam mı?\nErtesi sabah Oymakbeyi çocukları toplayıp sordu:\n-Söyleyin bakalım... Dün ne gibi bir iyilik yaptınız?\nBütün çocuklar, hep bir ağızdan:\n-Yaşlı bir kadının karşıdan karşıya geçmesine yardım ettik efendim.\nAdamcağız şaşırdı:\n-Hepiniz mi?\n-Evet efendim, hepimiz birden.\n-Neden?\nÇocuklardan biri cevap verdi:\n-Kadın karşıdan karşıya geçmek istemiyordu, ondan efendim!"
    },
    {
        "prompt": "Kadın kırmızı ışıkta geçerken polise ters cevap verir.",
        "story": "Yeni ilçe olan bir köye trafik ışıkları yeni konmuş, ışıkların altında bir polis bekliyor ve halkın ışıklara uymasını sağlamaya yani bir çeşit trafik eğitimi vermeye çalışıyormuş.\nO sırada, bakmış ki; bir kadın, elinde tuttuğu çocuğuyla, kırmızı yanarken karşıya geçiyor. Hemen seslenmiş:\n-Hanım, hanım! Nereye?\nKadın dönüp:\n-Vıy! demiş. Sana ne? Eltimgile gidiyom."
    },
    {
        "prompt": "Çocuk dövüşür, annesi pasta çörekle barış önerir.",
        "story": "Çocuk, okuldan bir gözü şiş olarak dönünce, annesi telaşlandı:\n-Oğlum ne oldu gözüne? Düştün mü yoksa?\n-Hayır düşmedim. Arkadaşım Orhan'la dövüştük. Ben de yarın onun gözünü şişireceğim!\nAnnesi yatıştırmaya çalıştı:\n-Sakın ha! Dövüşmek iyi bir şey değil. Ben sana yarın pasta çörek vereyim. Arkadaşına da ver, barışın. Güzel güzel oynayın olmaz mı?\n-Olur anneciğim, barışırız.\nErtesi gün, çocuk öteki gözü de şişmiş olarak döndü. Annesi merakla sordu:\n-Yine ne oldu?\n-Arkadaşım yaptı, daha çok pasta, çörek istiyor!"
    },
    {
        "prompt": "Spiker sınavında kravat takmadığı için alınmaz.",
        "story": "-Hayrola nereden?\n-Be be ben mi? Rad rad radyodan geliyorum...\n-Ne vardı radyoda?\n-Spi spi spi spiker sı sı sınavı vardı da...\n-Eeee, ne oldu?\n-Bı bı bı bırak yahu? Kı kı kıravat tak tak takmadık diye almadılar."
    },
    {
        "prompt": "Leyla boş kağıda keçi resmi çizer.",
        "story": "Leyla, ağacın altına oturmuş resim yapıyordu. Babası kızın elindeki bomboş kağıdı görünce sordu:\n-Leyla, ne resmi yapıyorsun bakayım?\n-Çimenlikte bir keçi resmi.\n-Çimenler nerede?\n-Keçi hepsini yedi.\n-Ya keçi?\n-Yiyecek bir şey kalmayınca o da gitti."
    },
    {
        "prompt": "Müşteri otelde ölü pireden şikayet eder.",
        "story": "Otelciyi çağırdı:\n-Odalarım temizdir, dediniz. Pire filan yoktur, dediniz. Bakın şuna!\nOtelci eğilip baktı:\n-Evet, o pire ama... Ölü... Ölü...\nMüşteri boynunu büktü, otelci de gitti.\nErtesi sabah otelci:\n-Nasıl, dedi, rahat uyudunuz mu?\n-Valla uyuyabilseydim, belki rahat ederdim ama... Sizin o ölü pire yok mu?\n-N'olmuş ölü pireye?\n-Yoo... Siz haklıymışsınız... Gerçekten ölüymüş o pire... Fakat cenaze töreni o kadar kalabalık oldu ki... Eşi, dostu, ahbabı, akrabası, bütün pireler hazırdı törende..."
    },
    {
        "prompt": "İki odacı, müdürlerinden daha aptal olduğunu düşünür.",
        "story": "İki müdür odacılarından şikayet ediyormuş. İkisi de kendi odacısının daha aptal olduğunu iddia ediyormuş. Birbirlerine kabul ettirmeye bahse girmişler.\nBir tanesi zile basarak odacısını çağırmış ve demiş ki:\n-Al şu 50 bin lirayı, git bana en son model arabalardan birini al getir.\n-Başüstüne, diyerek çıkmış odacı kapıdan.\nArdından diğer müdür kendi odacısını çağırmış:\n-Git şimdi bizim eve bak bakalım ben evde miyim?\n-Emredersiniz efendim, demiş ikinci odacı.\nTam kapıda iki odacı karşılaşmışlar, onlar da müdürlerini çekiştirmeye başlamışlar.\nBiri demiş ki:\n-Ya şu bizim müdür de çok salak, bana 50 bin lira verdi, git son model bir araba al gel dedi. Bugün pazar hiçbir yer açık değil ki!\n-O da bir şey mi, asıl benimki benden evine gidip kendisinin evde olup olmadığını öğrenmemi istedi. Aptal adam, elinin altında telefon var açıp sorsana!"
    },
    {
        "prompt": "Dilenci körlüğü denemiş ama yüzlük diye ellilik vermişler.",
        "story": "Adam, köşe başındaki dilenciye para verirken gönlünü de almak istedi:\n-Ayağın topal ama şükret, ya kör olsaydın?\n-Körlüğü de denedim be abi, iş yok! Yüzlük diye ellilikleri yutturuyolar..."
    },
    {
        "prompt": "Vali halkın en sevdiği valiyi sorar, cevap şaşırtır.",
        "story": "Adamın biri Erzurum'a vali tayin edilmiş. Gitmiş, görevi devralmış.\nHalkı ve çevreyi tanımak için çıktığı gezilerin birinde köy halkına sormuş:\n-Şimdiye kadar Erzurum'a tayin edilmiş valiler içinde size en çok hizmet eden hangisiydi?\nKöylünün biri cevap vermiş:\n-Sizden iki önceki valiydi; Mehmet Paşa.\n-Yaaaa, öyle mi, peki size ne gibi hizmetler yaptı?\n-Daha Erzurum'a gelirken, yolda, Bayburt'ta öldü!"
    },
    {
    "prompt": "Akıl hastanesinde zeytin ve hamam böceği testi yapılır.",
    "story": "Akil hastanesinden iki deliyi salivereceklermis. Doktorlar kendi aralarinda 'Sunlara son bir test yapalim da gorelim akillari baslarina gelmis mi.' demisler. Bunun uzerine iki deliyi bir masa basina cagirmislar. Masanin uzerine bir kavanoz dolusu siyah zeytin, bir kavanoz dolusu da canli hamambocegi dokmusler ve 'Buyrun beyler, yiyiniz.' demisler. Delilerden bir tanesi hemen zeytinlere saldirmis, otekisi araya girmis, 'Once kacanlari yiyelim, oburleri nasil olsa duruyor!'"
  },
  {
    "prompt": "İki deli bekçiden saklanır, biri miyavlar diğeri de onu tekrarlar.",
    "story": "Akıl hastanesinden kaçan iki deli, karşıdan gelen bekçiyi görünce iri gövdeli bir çınarın arkasına saklandılar. Bekçi, onların ayak seslerini işitmişti. Sordu:\n- Kim o?\nİçlerinden biri kedi gibi miyavladı. Bu başarılı miyavlamadan sonra bekçi yürüyüp gidiyordu ki, delilerin ayakları altındaki yapraklar hışırdadı. Bekçi geri dönüp yine seslendi:\n- Kim var orada?\nİkinci deli cevap verdi:\n- Bir kedi daha."
  },
  {
    "prompt": "İki deli direksiyon bulup yolculuğa çıkar.",
    "story": "İki deli, yolda giderken bir direksiyon bulunca çok sevindiler. O sevinçle saatte 160’la uzunca bir süre yol aldıktan sonra benzincinin önünde durdular. Arabayı süren;\n- Yüz bin liralık dedi. Süper olsun. Benzinci ikisini de tepeden tırnağa süzdükten sonra ;\n- Gidin işinize be! diye bağırdı. Sizin civatalarınız gevşek!\nİkincisi, araba kullanana döndü\n- Gördün mü! Araba masraf kapısı açtı bile!"
  },
  {
    "prompt": "Deli sekiz şekerli çay ister, hepsi eriyor der.",
    "story": "Deli, kahveye girdiğinde soluk soluğaydı. Boş bir masaya oturup ocağa seslendi;\n- Bana bir çay!\nÇay geldi, şekerleri atıp karıştırdı. Garsondan yine şeker istedi. Onları da atıp karıştırdı, yeniden istedi. Garson;\n- Sekiz şeker koydun çaya, dedi şaşkın şaşkın.\n- Koydum ama, işte görüyorsun, hepsi eriyor!"
  },
  {
    "prompt": "Deli oltayla balık değil, alık tuttuğunu söyler.",
    "story": "Deli duvara oturmuş. Elindeki oltanın ucu sokağa sarkmış.... Yoldan geçen soruyor;\n- Orada balık mı tutuyorsun sen?\n- Hayır alık tutuyorum.\n- Tutabildin mi bari ?\n- Çook ... Seninle 23 oldu!"
  },
  {
    "prompt": "Deli gişeden defalarca bilet alır.",
    "story": "Sinemaya girmek istiyordu. Gişeden biletini aldı. Birkaç dakika sonra gelip bir tane daha aldı. Sonra bir bilet daha bir daha... Gişedeki görevli dayanamadı\n- Karaborsa yapıyorsun galiba. Bu kaçıncı bilet alışın?\nDeli ;\n- İçeride bir deli var, dedi. Tam kapıdan girince biletimi yırtıyor. Ben de gelip yenisini almak zorunda kalıyorum.."
  },
  {
    "prompt": "İki deli gökkuşağını görünce yorum yapar.",
    "story": "İki deli, yağmurdan sonra kumaşı yırtık paslı bir şemsiyeyi açmışlar yolda gidiyorlardı. Birincisi, gökkuşağını gösterdi;\n- Bak bak ..\nİkinci baktı ve birden sinirlendi;\n- Hükümet böyle şeyler için para harcıyor da, bizim gibi deliler için doğru dürüst bir hastane yaptırmıyor!"
  },
  {
    "prompt": "Adam delilerin deli deli sıraya girip baktığını merak eder.",
    "story": "Adamin biri deliler hastanesini gezmeye gitmiş. Bakmış deliler kapıdaki delikten içeri doğru bakıyorlar. Bakan tekrar sıraya geçiyor. Devamlı bir döngü gibi olay yineleniyor. Adam merak etmiş, o da sıraya girmiş. Sıra kendisine gelmiş. Eğilip bakmış. Zifiri karanlık hiçbir şey yok. Bir tanesini durdurup sormuş:\n-Yahu ben hiç bir şey göremedim? Deli şaşırmış:\n-Ulan biz iki yıldır bakıyoruz bir şey göremiyoruz. Sen ilk bakışta mı göreceksin."
  },
  {
    "prompt": "Akıl hastası karşı kaldırımdaki yerini sorar.",
    "story": "Bir akıl hastası, bulunduğu kaldırımdan karşıya geçip rastladığı ilk görevliye sormuş :\n-Affedersiniz, karşı kaldırım nerede acaba?\nGörevli şaşırmış ama yine de karşı tarafı göstererek :\n-İşte şurada, demiş.\n-Kime yutturuyorsun yahu... Daha şimdi orda sordum, burayı gösterdiler!..."
  },
  {
    "prompt": "Akıl hastası mektup yazıyor ama ne yazdığını bilmez.",
    "story": "Akıl hastanesinde koğuşları gezen başhekim, bir hastanın oturmuş, bir şeyler yazdığını gördü :\n-Kolay gelsin, ne yazıyorsun?\n-Mektup yazıyorum efendim.\n-Yaaa... Kime yazıyorsun?\n-Kendime...\n-Peki, ne yazılı mektupta?\n-İlahi doktor bey, deli misiniz siz? Mektubu daha almadım ki... İçinde ne yazdığını bileyim."
  },
   {
    "prompt": "Yalnızken kendi kendine konuşma huyu var mı?",
    "story": "Arkadaşı Karadenizliye sormuş:\n-Yalnızken kendi kendine konuşma huyun var mıdır?\n-Ben kendi kendime konumam, demiş Karadenizli. Adamı gözümün önüne getiririm, öyle konuşurum."
  },
  {
    "prompt": "Temel ormanın güzelliğini gösterir, Dursun ağaçları göremez.",
    "story": "Temelle Dursun ormanda yürüyorlar. Bir ara Temel Dursun'a sesleniyor:\n-Dursun ormanın güzelliğine bak.\nDursun:\n-Ağaçlardan göremiyorum ki."
  },
  {
    "prompt": "Karadenizli güneşe gitmeyi planlar, akşam serinliğinde gitmeyi düşünür.",
    "story": "Bir mecliste konuşulurken,\nAmerikalı :\n-Biz Mars'a gideceğiz, demiş.\nAlman :\n-Biz yakıtsız giden otomobil üreteceğiz, demiş.\nFransız :\n-Atom bombasını etkisiz hale getirecek projelerimiz var, demiş.\nBizim Karadenizli de onlardan geri kalmamak için :\n-Biz de güneşe gideceğiz, demiş.\n-Güneşe gidemezsiniz, demişler. Güneş yakar.\nKaradenizli gülümsemiş :\n-O kadar da enayi değiliz, tabi, demiş. Akşam serinliğinde gideceğiz."
  },
  {
    "prompt": "Fadime kürk giyince hasta olur, Temel alerjiden şüphelenir.",
    "story": "Temel, Cemal'e :\n-Fadime'nin kürke alerjisi var.\n-Nerden pileysun?\n-Ne zaman kürk giymiş pi avrat cörse hastalanayı."
  },
  {
    "prompt": "Karadenizli otobüste anlamlı bakışlarla iletişim kurar.",
    "story": "Karadenizlinin biri hemşerisine anlatıyor :\n-Dün belediye otobüsüne bindim; yan koltuktaki adam bilet almamışım gibi bana anlamlı anlamlı baktı.\n-Sen ne yaptın?\n-Bende bilet almışım gibi anlamlı anlamlı ona baktım."
  },
  {
    "prompt": "Temel kırtasiyeden roman ister, ağırlığı sorun olmaz.",
    "story": "Temel kırtasiye'ye girmiş, tezgahtara :\n-Pana pir roman lazum, demiş.\nKırtasiye tezgahtarı sormuş :\n-Efendim agır mı olsun hafif mi?\nTemel :\n-Farketmez, nasul olsa arabam dısarudadur."
  },
  {
    "prompt": "Temel'e rüyasında Allah yürü ya kulum der, o da arabasını satar.",
    "story": "Temel'e rüyasında Allah yürü ya kulum demiş. Temel de arabasını satmış."
  },
  {
    "prompt": "Temel beş kere beş sorusuna farklı cevaplar alır.",
    "story": "Aritmetik öğretmeni Temel öğrencilerinden şikayet ediyormuş :\n-Derste peş kere peş kaç ediy, diye sorayrum, kırk cevapı alayrum. Halbuki peş kere peş yirmi peş, pilemedun otuz."
  },
  {
    "prompt": "Karadenizliye eşek denilince karşılık verir.",
    "story": "Adamın biri Karadenizli arkadaşına \"eşek\" demiş.\nKaradenizli sormuş :\n-Eşek olduğum için mi arkadaşınım; yoksa arkadaşın olduğum için mi eşeğim?"
  },
  {
    "prompt": "Temel asansör bozulunca alternatif önerir.",
    "story": "Temel kapıcı, çalıştığı on katlı binanın asansörü bozulunca bir kağıt asıyor, üstünde şu yazılar var :\n-Asansör pozuk, en yakın asansör yüz metre ileride, yandaki pinadadur."
  },
  {
    "prompt": "Temel hayvanat bahçesinde açık kafese girer.",
    "story": "Temel hayvanat bahçesinde gezerken açık bulduğu bir kafesten içeri dalmış.\n-Hoop, dur ne yapıyorsun, orası aslan kafesi, diye bağırışmışlar.\nTemel geri dönmüş,\n-Sankim aslanınızı yedük, demiş."
  },
  {
    "prompt": "Temel kendini ağaca belinden asar, Dursun kurtarır.",
    "story": "Dursun evinden çıktığında bir de bakar ki komşusu Temel kendini belinden ağaca asmış halde duruyor.\nHemen gidip ipi ağaçtan çözer. Komşusunu ağaçtan indirdikten sonra merakla sorar :\n-Ha sen ne yapayudun öyle?\n-Hiç kendimi asaydum...\n-Ha uşağum, penum pildiğum insan poynundan asılayi.\nTemel üzgün ve çaresiz bir halde komşusu Dursun'a baktıktan sonra cevap verir :\n-Ben de öyle yapmişudum. Ama ipu poynima pağladığum zaman bi türlü nefes alamayrum."
  },
  {
    "prompt": "Karadenizli yalnızken kendi kendine konuşur mu?",
    "story": "Arkadaşı Karadenizliye sormuş:\n-Yalnızken kendi kendine konuşma huyun var mıdır?\n-Ben kendi kendime konumam, demiş karadenizli. Adamı gözümün önüne getiririm, öyle konuşurum."
  },
  {
    "prompt": "Temel ormanda doğayı gösterir ama Dursun ağaçları görür.",
    "story": "Temelle Dursun ormanda yürüyorlar. Bir ara Temel Dursun’a sesleniyor:\n-Dursun ormanın güzelliğine bak.\nDursun:\n-Ağaçlardan göremiyorumki."
  },
  {
    "prompt": "Karadenizli güneşe gitmek ister, akşam serinliğini bekler.",
    "story": "Bir mecliste konuşulurken,\nAmerikalı:\n-Biz Mars'a gideceğiz, demiş.\nAlman:\n-Biz yakıtsız giden otomobil üreteceğiz, demiş.\nFransız:\n-Atom bombasını etkisiz hale getirecek projelerimiz var, demiş.\nBizim Karadenizli de onlardan geri kalmamak için:\n-Biz de güneşe gideceğiz, demiş.\n-Güneşe gidemezsiniz, demişler. Güneş yakar.\nKaradenizli gülümsemiş:\n-O kadar da enayi değiliz, tabi, demiş. Akşam serinliğinde gideceğiz."
  },
  {
    "prompt": "Temel, Fadime'nin kürke alerjisi olduğunu söyler.",
    "story": "Temel, Cemal'e:\n-Fadime'nin kürke alerjisi var.\n-Nerden pileysun?\n-Ne zaman kürk giymiş pi avrat cörse hastalanayı."
  },
  {
    "prompt": "Karadenizli bilet almadıysa da bakışlarıyla durumu kurtarır.",
    "story": "Karadenizlinin biri hemşerisine anlatıyor:\n-Dün belediye otobüsüne bindim; yan koltuktaki adam bilet almamışım gibi bana anlamlı anlamlı baktı.\n-Sen ne yaptın?\n-Bende bilet almışım gibi anlamlı anlamlı ona baktım."
  },
  {
    "prompt": "Temel kırtasiyeye roman almaya gider.",
    "story": "Temel kırtasiye'ye girmiş, tezgahtara:\n-Pana pir roman lazum, demiş.\nKırtasiye tezgahtarı sormuş:\n-Efendim agır mı olsun hafif mi?\nTemel:\n-Farketmez, nasul olsa arabam dısarudadur."
  },
  {
    "prompt": "Temel rüyasında Allah'tan mesaj alır ve arabasını satar.",
    "story": "Temel'e rüyasında Allah yürü ya kulum demiş. Temel de arabasını satmış."
  },
  {
    "prompt": "Temel çarpım tablosunda karışıklık yaşar.",
    "story": "Aritmetik öğretmeni Temel öğrencilerinden şikayet ediyormuş:\n-Derste peş kere peş kaç ediy, diye sorayrum, kırk cevapı alayrum. Halbuki peş kere peş yirmi peş, pilemedun otuz."
  },
  {
    "prompt": "Karadenizli arkadaşı hakaret mi etti yoksa arkadaşlık mı etti?",
    "story": "Adamın biri karadenizli arkadaşına 'eşek' demiş.\nKaradenizli sormuş:\n-Eşek olduğum için mi arkadaşınım; yoksa arkadaşın olduğum için mi eşeğim?"
  },
  {
    "prompt": "Temel asansör arızasına çözüm getirir.",
    "story": "Temel kapıcı, çalıştığı on katlı binanın asansörü bozulunca bir kağıt asıyor, üstünde şu yazılar var:\n-Asansör pozuk, en yakın asansör yüz metre ileride, yandaki pinadadur."
  },
  {
    "prompt": "Temel aslan kafesine girer.",
    "story": "Temel hayvanat bahçesinde gezerken açık bulduğu bir kafesten içeri dalmış.\n-Hoop, dur ne yapıyorsun, orası aslan kafesi, diye bağırışmışlar.\nTemel geri dönmüş:\n-Sankim aslanınızı yedük, demiş."
  },
  {
    "prompt": "Temel kendini belinden ağaca asar, Dursun müdahale eder.",
    "story": "Dursun evinden çıktığında birde bakar ki komşusu Temel kendini belinden ağaca asmış halde duruyor. Hemen gidip ipi ağaçtan çözer. Komşusunu ağaçtan indirdikten sonra merakla sorar:\n-Ha sen ne yapayudun öyle?\n-Hiç kendimi asaydum...\n-Ha uşağum, penum pildiğum insan poynundan asılayi.\nTemel üzgün ve çaresiz bir halde komşusu Dursun'a baktıktan sonra cevap verir:\n-Ben de öyle yapmişudum. Ama ipu poynima pağladığum zaman bi türlü nefes alamayrum."
  },
   {
    "prompt": "Delinin taburcu edilmeden önce yaptığı zeka testi.",
    "story": "Delinin birisi hastaneden taburcu olacakmış ve son muayene için baş hekim gelir. Deliye sorar:\n-Elin nerede?\nDeli gösterir.\n-Bacağın nerede?\nDeli yine gösterir.\n-Burnun nerde?\nDeli yine gösterir.\nBaş hekim doktorlara:\n-Bırakın emrini verir ve çıkar. Hekim çıktıktan sonra deli göbeğini gösterir ve:\n-Bende bu kafa varken tabi salıverirsiniz, der."
  },
  {
    "prompt": "İki deli saat hakkında konuşur.",
    "story": "İki deli arasında konuşma:\n-Saat kaç?\n-Beş var\n-Kaça beş var?\n-Bilmiyorum, akrebini kaybettim."
  },
  {
    "prompt": "Deliler bekçiye kedi gibi seslenerek saklanır.",
    "story": "Akıl hastanesinden kaçan iki deli, karşıdan gelen bekçiyi görünce iri gövdeli bir çınarın arkasına saklandılar. Bekçi, onların ayak seslerini işitmişti. Sordu:\n-Kim o?\nİçlerinden biri kedi gibi miyavladı. Bu başarılı miyavlamadan sonra bekçi yürüyüp gidiyordu ki, delilerin ayakları altındaki yapraklar hışırdadı. Bekçi geri dönüp yine seslendi:\n-Kim var orada?\nİkinci deli cevap verdi:\n-Bir kedi daha."
  },
  {
    "prompt": "Müdür delikten dışarı bakar ama bir şey göremez.",
    "story": "Akıl hastanesine yeni atanan müdür hastaneyi dolaşmaya karar vermiş. Dolaşırken hastanesinin dışarıya bakan duvarının dibinde bir grup akıl hastasının tek sıra olup duvardaki bir delikten baktıklarını görmüş. Merak içinde yanlarına giderek:\n-Yahu hepiniz toplanmış burada ne yapıyorsunuz.\n-Hiçbir şey yapmıyoruz sadece bu delikten dışarı bakıyoruz...\nBunun üzerine müdür hastaları kenara iterek:\n-Durun birde ben bakayım, demiş ve delikten dışarıya doğru bakmış. Birde ne görsün delik kapalı ve hiçbir şey görünmüyor. Hiddetle akıl hastalarına dönerek:\n-Yahu, demiş, Ben baktım bu delikten dışarı bir şey görünmüyor peki siz ne görüyorsunuz:\n-Deliler hep bir ağızdan Müdür Bey, demiş. Biz yıllardan beri bakıyoruz bir şey göremedik siz bir bakışta nasıl göreceksiniz ki."
  },
  {
    "prompt": "Deli gazeteci olur, diğerleri eski gazetedir.",
    "story": "Başhekim, akıl hastanesinin bahçesinde dolaşıyordu, bir ara baktı, bir kalabalık gözüne çarpmıştı. Hemen oraya seğirtti. Deliler bir halka oluşturmuş, ortada dönüp konuşan birini dinliyorlardı:\n-Papendreu seçimleri kaybetti. Hastaneye kaldırıldı... Bulgar zulmü devam ediyor. Zorla yollanan soydaşlarımızın sayısı seksen bine ulaştı... Federasyon kupasını Beşiktaş kazandı...\nBaşhekim bu işten hoşlanmış:\n-Ne yapıyorlar bunlar böyle? diye sormuş.\n-Efendim, demişler. Ortadaki deli kendinin gazete olduğunu sanıyor, haberleri bildiriyor.\nBaşhekim daha da hoşlanmış. Dolaşmasını sürdürmüş. Az ileride bir de ne görsün! Sekiz, on deli iplerle sımsıkı birbirlerine bağlanıp bir köşeye atılmamış mı!\n-Onlar mı, okunup da iadeye gidecek eski gazeteler efendim..."
  },
  {
    "prompt": "Deli kuyudaki taşı attı, akıllılar çıkaramıyor.",
    "story": "Delinin biri kuyuya bir taş atmış yüz akıllı çıkarmaya çalışmış, çıkaramamış. Sonunda delinin diğeri ilk deliye bu akıllıların ne yaptığını sormuş. Birinci deli de:\n-Elimdeki taşı kuyudan çıkarmaya çalışıyorlar, demiş."
  },
  {
    "prompt": "Pamuklu çorap tercihi yüzünden hastanede.",
    "story": "Akıl hastanesinde doktor, davranışlarını normal bulduğu hastaya niçin hastanede bulunduğunu sorar.\nHasta:\n-Pamuklu çorapları yünlülere tercih ettiğim için, diye cevap verir.\nŞaşıran doktor:\n-Bunun anormallik neresinde? Ben de pamuklu çorapları tercih ederim, der.\nHasta sevinçle karşılık verir:\n-Çok memnun oldum doktor. Sizinkiler limonlu mu, yoksa sirkeli mi?"
  },
  {
    "prompt": "Deli çiviyi ters çakar, başka deli sebebini açıklar.",
    "story": "Delinin biri, çiviyi tersine çevirerek sivri tarafına vura vura duvara çakmaya başlamış.\nOnun bu halini gören başka bir deli işe karışmış:\n-Baksana, yahu! Sen yanlış bir iş görüyorsun. Bu çivi karşıki duvarın çivisi olacak galiba, demiş."
  },
  {
    "prompt": "Akıl hastası karşı kaldırımın yerini sorar.",
    "story": "Bir akıl hastası, bulunduğu kaldırımdan karşıya geçip rastladığı ilk görevliye sormuş:\n-Affedersiniz, karşı kaldırım nerede acaba?\nGörevli şaşırmış ama yine de karşı tarafı göstererek:\n-İşte şurada, demiş.\n-Kime yutturuyorsun yahu... Daha şimdi orda sordum, burayı gösterdiler!..."
  },
  {
    "prompt": "Deli kendine mektup yazıyor.",
    "story": "Akıl hastanesinde koğuşları gezen başhekim, bir hastanın oturmuş, birşeyler yazdığını gördü:\n-Kolay gelsin, ne yazıyorsun?\n-Mektup yazıyorum efendim.\n-Yaaa...Kime yazıyorsun?\n-Kendime...\n-Peki, ne yazılı mektupta?\n-İlahi doktor bey, deli misiniz siz? Mektubu daha almadım ki... İçinde ne yazdığını bileyim."
  },
  {
    "prompt": "Pilot başhekimi kandırıp uçar.",
    "story": "Uçak, Yeşilköy'den kalkmıştı. Bakırköy Akıl Hastanesinin üzerinden geçerken, pilot birden gülmeye başladı. Hostes bu gülüşün sebebini sorunca şu cevabı verdi:\n-Başhekim kaçtığımı öğrenince kimbilir nasıl şaşıracak!!!"
  },
  {
    "prompt": "Ağaçtaki deliler armut, yerdeki olgunlaşmış.",
    "story": "Bir müfettiş akıl hastanesini geziyormuş. Bahçeye gelince delilerin ağaçta asıldığını ama birinin yere yattığını görünce yatana sormuş:\n-Neden ağaca çıktılar, demiş.\nO da:\n-Armut sanıyolar kendilerini, demiş.\nMüfettiş:\n-Sen armut değil misin?, demiş.\nO da hayır ben olgunlaşıp yere düştüm demiş."
  },
  {
    "prompt": "Katil olduğunu söyleyen deli suçsuz bulunur.",
    "story": "Katil, suçunu itiraf etti, yargıç da durumu jüri heyetine iletti. Biraz sonra jüri başkanı kararı açıkladı:\n-Bu sanık suçsuzdur...\nYargıç adamakıllı kızdı:\n-Canım, ne biçim iş bu!... Adam, ben katilim diyor suçunu itiraf ediyor siz de suçsuzdur kararına varıyorsunuz... Acaba, suçsuzdur kararını neye dayanarak verdiniz?\n-Delilik efendim, delilik...\nYargıç bütün jüri üyelerini teker teker süzdü. Başını sallayarak:\n-Sahi mi? 12'niz de mi?.."
  },
  {
    "prompt": "Deli önce kaçan hamamböceğini yemek ister.",
    "story": "Akıl hastanesinden iki deliyi salıvereceklermiş. Doktorlar kendi aralarında:\n-Şunlara son bir test yapalım da görelim akılları başlarına gelmiş mi, demişler. Bunun üzerine iki deliyi bir masa başına çağırmışlar. Masanın üzerine bir kavanoz dolusu siyah zeytin, bir kavanoz dolusu da canlı hamamböceği dökmüşler ve:\n-Buyrun beyler, yiyiniz, demişler. Delilerden bir tanesi hemen zeytinlere saldırmış, ötekisi araya girmiş.\n-Önce kaçanları yiyelim, öbürleri nasıl olsa duruyor!"
  },
  {
    "prompt": "Kapı açık olunca kaçış planı iptal olur.",
    "story": "Akıl hastanesinde deliler bir araya gelip kaçış planı yaparlar. Elebaşları planı anlatır:\n-Büyük bir kütük bulup ilk önce 1. kapıyı, 2. kapıyı ve daha sonra 3. kapıyı kıracağız ve herkes başının çaresine bakıp kaçacak. Sabah olunca bir kütük bulurlar doğruca 1. kapıyı kırarlar, 2. kapıya koşup onu da kırdıktan sonra 3. kapıya yönelirler. 3. kapının açık olduğunu gören elebaşları der ki:\n-Arkadaşlar plan bozuldu geri dönün."
  },
  {
    "prompt": "Deli saatini havuza atar, yüzmesini izlemek ister.",
    "story": "Deli, saatini hastane bahçesindeki havuza atmıştı. Bunu gören arkadaşı:\n-Niye attın saati havuza, dedi.\n-Nasıl yüzdüğünü görmek için.\n-Peki, kurdun mu?\n-Hayır.\n-Enayi, kurmadan yüzer mi?"
  },
  {
    "prompt": "Kedi yıkandıktan sonra kurutulunca ölür.",
    "story": "İki arkadaşın, bir kedisi varmış. Birisi:\n-Zavallı kedi çok kirlenmiş ben onu yıkayayım, demiş. Diğer arkadaşı:\n-Hayır yıkama yoksa ölür, demiş. Bizimki dinlememiş ve kediyi yıkamış ve kedi ölmüş. Arkadaşı:\n-Ben sana demedim mi kedi ölür diye, demiş. Cevap şu:\n-Ama ben kediyi yıkarken ölmedi, sıkarken öldü."
  },
  {
    "prompt": "Deli kendini Tanrıoğlu olarak tanıtır, yaşlı hasta yalanlar.",
    "story": "Akıl hastanesine yeni gelen doktor, hastaları ziyaret ediyordu. Birine yaklaştı:\n-Sizin adınız nedir bakayım?\n-Hüsamettin efendim.\n-Soyadınız?\n-Tanrıoğlu.\nTam o sırada yandaki yaşlı:\n-İnanma inanma doktor, yalan söylüyor. Benim böyle bir oğlum yoktur."
  },
  {
    "prompt": "Havuz boş ama deli yine de atlayacak.",
    "story": "Mühim bir şahsiyet, bir akıl hastalığı kliniğini gezerken delilerin bahçedeki havuza atladıklarını görür ve başhekime dönerek:\n-Mükemmel, hastalarınızın her türlü ihtiyacını karşıladığınızı görüyorum. Başhekim teşekkür eder, sonra da sözlerine devam eder:\n-Hele siz bir de su doldurabildiğimiz zaman gelin de görün!\nHavuzun boş olduğunu öğrenen adamcağız dehşet içinde tramplenin altına koşar ve heyecanla atlamaya hazırlanan deliye 'atlamamasını, havuzun içinde su olmadığını' söyler. Deli:\n-Ne zararı var? Zaten ben de yüzme bilmiyorum ki!"
  },
  {
    "prompt": "Deliler uçakta gürültü yapar, sonra birden kaybolur.",
    "story": "Delileri uçağa bindirmişler, bir şehirden ötekine naklediliyorlardı. Ama o kadar çok gürültü yapıyorlardı ki, sonunda pilot dayanamadı, uçağı ikinci pilota teslim ederek içeride ne olup bittiğini görmek istedi.\nDeliler uçakta hep bir ağızdan bağırıp çağırıyorlardı. Baktı, en başta, bir deli, ötekilere uymamış, akıllı, uslu oturuyordu.\n-Sen neden bağırmıyorsun? diye soracak oldu.\nAdam:\n-Ben bunların öğretmeniyim, diye cevap verdi. Onlar da benim öğrencilerim. Şimdi teneffüsteler de onun için ses çıkartmıyorum.\nPilot, çaresiz yerine döndü. Bir süre geçti. Bir an geldi ki sesler büsbütün kesiliverdi.\nPilot:\n-Aman çok güzel! diye sevindi. Herhalde kendinin öğretmen olduğunu sanan deli, ötekileri derse almış olsa gerek, diye düşündü.\nAma dakikalar geçiyor, arkadan hiçbir ses seda çıkmıyordu. Pilot biraz daha bekledikten sonra merak etti. Gidip bakmak istedi.\nBir de ne görsün! Uçağın kapısı açık ve içeride öğretmenden başka kimsecikler yok!\nDehşetle sordu:\n-Öğrencilerin nerede?, diye...\n-Dersler bitti. Hepsini evlerine gönderdim!"
  },
  {
    "prompt": "Adam olmanın yöntemi nedir?",
    "story": "Günün birinde Hoca'nın da içinde bulunduğu topluluktan birisi; 'Hocam, adam olmanın yöntemi nedir?' deyince; Hoca Efendi, adamın nefes almasına bile fırsat vermeden; 'Canım, bunu bilmeyecek ne var, elbette kulaktır.' der. Fakat Hoca, arkadaşlarının 'kulaktır' cevabından pek bir şey anlamadıklarını anlayınca açıklama yapma gereğini duyar: 'Aa! Bunu bilemeyecek ne var? Herhangi bir adam konuşurken onu can kulağı ile dinlemeli; bu arada kendi ağzından çıkanı kendi kulağı duymalıdır.'"
  },
  {
    "prompt": "Allah’ın rahmetinden kaçılır mı?",
    "story": "Günün birinde bardaktan boşanırcasına yağmur yağmaktadır. Nasreddin Hoca pencereden bakarken bir komşusunun yağmurdan kaçtığını görür ve pencereyi açarak; 'Komşu, utanmıyor musun, niçin Allah’ın rahmetinden kaçıyorsun?' der. Komşu koşmayı bırakır ama sırılsıklam olur. Ertesi gün Hoca dışarıda hızlı adımlarla yürürken aynı komşusuyla karşılaşır. Komşu; 'Sen dün Allah’ın rahmetinden kaçılmaz demiştin, şimdi neden koşuyorsun?' diye sorar. Hoca gülümseyerek: 'Ben Allah’ın rahmetinden kaçmıyorum, Allah’ın rahmetini çiğnememek için koşuyorum.' der."
  },
  {
    "prompt": "Altın olsa ne, taş olsa ne?",
    "story": "Nasreddin Hoca yolculuk sırasında bir şehre uğrar. Orada bazı evlerin üzerinde bayrak olduğunu fark eder. Sorarlar: 'Hocam, o evlerde küp dolusu altın vardır.' Hoca da bir küp alıp içini çakıl taşlarıyla doldurur. Sohbet sırasında bir misafir küpte altın yerine taş olduğunu fark eder ve sorar: 'Bu nasıl iş?' Hoca cevap verir: 'Yahu komşular, küpte yattıktan sonra altın olsa ne, taş olsa ne? Fark eden ne ki?'"
  },
  {
    "prompt": "Eşeğin ayaklarını ikiye indirmek isteyen komşu.",
    "story": "Nasreddin Hoca’dan hoşlanmayan bir komşusu onu yolda durdurur: 'Senin için evliya oldu diyorlar, benim dört ayaklı eşeğimi iki ayaklı yap da inanayım.' Hoca sinirlenerek: 'Be adam, eşeğin ayaklarını dörtten ikiye indirebilir miyim bilmem, ama sen biraz daha konuşursan senin ayaklarını dörde çıkarabilirim.' der."
  },
  {
    "prompt": "Hoca ile kardeşi aynı yaşta mı?",
    "story": "Arkadaşları Hoca’ya takılır: 'Sen mi büyüksün, yoksa kardeşin mi?' Hoca cevap verir: 'Geçen yıl anneme sormuştum, kardeşin senden bir yaş küçük demişti. O zamandan beri bir yıl geçtiğine göre şimdi aynı yaştayız.'"
  },
  {
    "prompt": "İnsanlar neden hep şikayet eder?",
    "story": "Hoca ve arkadaşları baharda bir çınarın altında otururken biri sorar: 'Yazın sıcaktan, kışın soğuktan şikayet ederler; bunun sebebi nedir?' Hoca hemen cevap verir: 'Komşu, sen onlara kulak asma. Bak içinde yaşadığımız bahardan hiç hoşnut olmayan var mı? Sen hayatını yaşamaya devam et.'"
  },
  {
    "prompt": "Pencerede başını unutan ev sahibi.",
    "story": "Hemşerileri bazen candan, bazen de sahte olarak Hoca’ya saygı gösterirler. Günün birinde sahte saygı gösterenlerden biri Hoca’yı evine davet eder. Hoca da konumu gereği davete gider. Gider gitmesine de eve yaklaşınca ev sahibinin başını pencereden içeriye doğru çektiğini görür.\nHiçbir şey olmamış gibi evin kapısına çalan Hoca:\n“Komşu, komşu ben geldim.” deyince, kapının arkasından değiştirilmiş bir ses duyulur:\n“Ah Hocam, ah! Evin sahibi buradaydı, az önce gitti, ben sizin geldiğinizi söylerim, mutlaka çok üzülecektir.”\nHoca bu söz karşısında iyice sinirlenir ve:\n“Ev sahibine söyleyin, bir daha bir yere giderken başını pencerede unutmasın.” der."
  },
  {
    "prompt": "Kavga eden kadınlar belki de barışmıştır.",
    "story": "Nasreddin Hoca evinin bahçesindeki ağacın gölgesinde namaz saatini beklerken telaşlı bir şekilde kapısının tokmağına vurulduğunu işitir. Hoca, kapıyı açınca komşusunu görür ve:\n“Buyur komşu, nedir bu telaşın?” deyince komşusu:\n“Sorma Hocam, karımla baldızım saç saça, baş başa dövüşüyorlar.” der.\nBunun üzerine Hoca merakla:\n“Komşu, ayıramadın mı?” deyince, komşusu sızlanarak cevap verir:\n“Ne mümkün Hocam, bırak ayırmayı yanlarına bile yaklaşamadım.”\n“Pekiyi, bu hanımlar ne diye kavga ediyorlar?” deyince komşusu:\n“Bilmiyorum Hocam!” der.\nHoca bir defa daha sorar:\n“Sakın, ‘sen yaşlısın, ben yaşlıyım’ diye kavga etmesinler?” deyince komşusu:\n“Yok Hocam, yok başka bir konuda kavga ediyor olmalılar!” der.\nBunun üzerine Hoca rahat bir şekilde konuyu çözüverir:\n“Komşum, o zaman telaşlanmaya gerek yok! Konu yaş değilse çabucak barışırlar, belki de şimdiye barışmışlardır bile.” der."
  },
  {
    "prompt": "Nal çakma sesiyle karıştırılan berber dükkanı.",
    "story": "Nasreddin Hoca tıraş olmak için berber koltuğuna oturduğunda ustanın olmadığını anlar, fakat iş işten de geçmiştir. Çünkü berber çırağı çoktan Hoca’yı tıraş etmeye başlamıştır bile. Berber çırağının hareketleri, aletleri kullanmadaki beceriksizliği artınca Hoca’nın da keyfi kaçar. Tam bu sırada komşu dükkândan garip garip sesler gelmez mi? Sanki orda bir öküz böğürüyor. Hoca, berberi biraz oyalamak için:\n“Bu ses nedir?” deyince berber çırağı:\n“Önemli bir şey değil, komşumuz nalbanttır; herhâlde öküze nal çakıyor.” der.\nBu sözleri işiten Hoca rahatlar:\n“Oh, çok şükür, ben de birisini tıraş ediyorlar sanmıştım.” der."
  },
  {
    "prompt": "Hoca’nın delikanlılığı ve eşeğe binememe hikayesi.",
    "story": "Günlerden bir gün Nasreddin Hoca, alışveriş yapmak için şehre gidecektir. Ahırdan eşeğini çıkarır, evin önüne getirir. Şehirden siparişi olan komşular Hoca’nın başına toplanırlar. Hoca, eşeğine binmeye çalışır, fakat her çaba boşunadır. Bir kez daha denemek ister \"Ha gayret” deyip bir daha eşeğin üstüne sıçrar ama bu kez de eşeğin üzerinden öbür tarafına düşüverir.\nKomşuları Hoca’nın gayretlerinin bu şekilde bitmesine bir taraftan üzülürler, bir taraftan da ellerinde olmadan gülmeye başlarlar.\nBu durum karşısında canı iyice sıkılan Hoca komşularına dönerek:\n“Yahu komşular, benim delikanlılığımı görmediniz. Ben, bir sıçrayışta değil eşeğe binmek damın üzerine bile atlardım.” der. Hoca, böyle der demesine de bir yandan da kendi kendine:\n“Hey gidi Hoca, ben senin delikanlılığını da bilirim.” deyiverir."
  },
  {
    "prompt": "Ölüm haberine Hoca’dan geçmişe dair ilginç cevap.",
    "story": "Nasreddin Hoca akşam üzeri evine gelince hanımının suratının asık olduğunu görür ve sorar:\n“Hanım, hayırdır, ne oldu sana?”\nHanım daha da üzgün bir tavırla cevap verir:\n“Daha ne olsun Hoca, sana söylemiştim ya!”\n“Neyi söylemiştin hanım, adamı meraklandırma!”\n“Biliyorsun ya, bizim komşu hastaydı...\"\n“Eee... Ne olmuş bizim komşuya?”\n“Sizlere ömür, komşu ölmüş!”\nHoca şöyle bir kafasını kaşıdıktan sonra:\n“Hanım, komşumuza Allah rahmet etsin; fakat ben senin düğün evinden gelişini de hatırlarım!” der."
  },
  {
    "prompt": "Eşekten düşen Hoca kendini kurtarır.",
    "story": "Günün birinde Hoca Efendi pazara gitmek için eşeğine biner ve yola koyulur. Bir süre gittikten sonra eşek huysuzlanır ve ardından hoplayıp zıplamaya başlar. Derken Nasreddin Hoca da eşekten düşüverir. Düşer düşmesine de çevresine toplanan çocuklar toplu hâlde bağırmaya başlarlar:\n“Nasreddin Hoca eşekten düştü, Nasreddin Hoca eşekten düştü.”\nHoca, şöyle bir sağına soluna baktıktan sonra büyüklerden kimselerin olmadığını görünce eşe dosta rezil olmamak için:\n“Çocuklar, eşekten düşmedim, ben zaten eşekten inecektim.” deyiverir."
  },
  {
    "prompt": "Tavşan yerine buğday çıkınca Hoca'nın kıvrak zekâsı.",
    "story": "Hoca, günün birinde başını alıp kırlara gezmeye çıkar. Epeyce dolaştıktan sonra nasıl olduysa önünden geçmekte olan bir tavşanı yakalar. Tavşanı hemen yanında bulunan heybenin gözüne koyar ve evine dönmeye karar verir. Hoca’nın amacı, tavşanı eşine dostuna gösterip onların tanıyıp tanımadıklarını öğrenmektir. Komşularına haber göndererek:\n“Bu akşam bize gelin, sizlere tuhaf bir yaratık göstereceğim.” der.\nHoca’nın hanımı da çok meraklı biridir. Heybeyi açar, fakat açmasıyla beraber tavşan heybenin gözünden zıplayarak kaçıverir. “Eyvah, Hoca buna çok kızacak!” diye düşünüp dururken aklına bir fikir gelir. Aceleyle karşısındaki rafta duran buğday tasını heybenin gözüne kor ve ağzını sıkıca bağlar.\nAkşam olur. Davetliler bir bir Hoca’nın evine gelirler. Herkes merakla bir şeyleri beklemeye koyulur. Derken Hoca, heybeyi eline alır, ağır aksak açmaya çalışır. Fakat bu sırada buğday ölçeği “Pat!” diye yere düşüvermesin mi? Herkesin birbirine şaşkın şaşkın baktığı bir anda Hoca, hemen söze girer ve:\n“İşte arkadaşlar; bilen var, bilmeyen var. Bunun on altısı bir kile eder!” deyiverir."
  },
  {
    "prompt": "Bu da Hoca’nın Atışı",
    "story": "Nasreddin Hoca, sağda solda ‘Ben şöyle yay çekerim, şöyle ok atarım.’ diye konuşur durur. Bunun gerçek olup olmadığını anlamak isteyen gençler onu yarışmaya davet ederler. Hoca, ilk okunu atar, ama hedefin çok uzağına düşer. Çevreden gülüşme sesleri artınca Hoca;\n‘Bu bizim subaşının atışı; o, oku böyle atar.’ der.\nİkinci olarak oku attığında, hedefi yine vuramaz, yine gülüşme sesleri arasında Hoca;\n‘Bu da bizim Kadı Efendi’nin atışı…’ der.\nÜçüncü olarak oku atan Hoca hedefi vurunca;\n‘Bu da Hoca’nın atışı…’ deyiverir."
  },
  {
    "prompt": "Bugünlerde Ay Alıp Satmadım",
    "story": "Nasreddin Hoca bir gün pazarda dolaşırken yanına bir adam yaklaşır ve;\n‘Hocam, bugün ayın kaçı?’ der.\nHoca, adamın niyetini anlamış olmalı ki;\n‘Arkadaş, bugünlerde hiç ay alıp satmadım, bilmem.’ cevabını verir."
  },
  {
    "prompt": "Bunlardan Daha İyi Bir Şahit Bulunabilir mi?",
    "story": "Nasreddin Hoca’nın Akşehir kadısı olduğu zamanda makamına bir adam girer. Bu adamın sıkıntısı olduğu hareketlerinden kolaylıkla sezilmektedir. Hoca;\n‘Anlat, bakalım, nedir derdin?’ dediğinde adam;\n‘Kadı Efendi benim bir tamburam vardı, onu filan adam çaldı, ondan davacıyım.’ der.\nKadının emri üzerine davalı huzura çağırılır. Kadı;\n‘Sen bu adamın tamburasını çaldın mı?’ deyince adam;\n‘Hayır, Kadı Efendi bu tambura benim babamdan kalmıştır, istersen şahitlerimi getirebilirim.’ der.\nKadı’nın emri üzerine şahitler davet edilir ve onlardan birincisine;\n‘Tamburanın sahibi kimdir?’ diye sorulduğunda şahit;\n‘Tambura dava edilen adamındır.’\nÖbür şahit de;\n‘Evet aynen öyledir, hatta ben tamburanın beş teli olduğunu bile biliyorum.’ der.\nŞahitlerin anlattıklarını duyan şikâyetçi ifadelere itiraz edince, Kadı Efendi sebebini sorar. Bu defa şikâyetçi adam;\n‘Kadı Efendi, şahitlerden birisi düğünlerin köçeği, birisi şarkıcısı, birisi de …’ deyince Kadı adamın sözünü keser ve;\n‘Be adam! Böyle bir davada bunlardan daha iyi bir şahit bulunabilir mi?’ deyiverir."
  },
  {
    "prompt": "Buralı Birine Soruver",
    "story": "Bir gün Nasreddin Hoca’nın yolu, daha önce hiç geçmediği bir köye düşer. Hoca’yı gören bir köylü;\n‘Efendi, Hoca’ya benziyorsun. Sen bilirsin, bugün günlerden ne?’\nKeyfi yerinde olmayan Hoca, köylüyü süzdükten sonra;\n‘Ben köyünüzün yabancısıyım. Ne bileyim sizin gününüzü? Sen buralı birine soruver.’ deyiverir."
  },
  {
    "prompt": "Buraya Yeni Taşındığımızı Sanıyordum",
    "story": "Hoca ile hanımı bir gece yataklarında mışıl mışıl uyurlarken evlerine hırsız girer. Usta hırsızın, onlar uyurlarken evde bulduğu değerli eşyaları bir çuvala doldurup kapıdan çıkacağı sırada Hoca Efendi uyanır. Bir bakar ki hırsız eşyalarını çuvala doldurmuş götürmektedir.\nAlelacele kalkan Hoca epeyce bir süre hırsızı takip ettikten sonra ikisi birlikte bir eve girerler. Bu ev de hırsızın evidir. Karşısında Hoca’yı gören hırsız heyecanlı bir şekilde;\n‘Hoca Efendi, benim evimde senin ne işin var? Burası benim evim, haydi var git işine!’ der.\nHoca, hırsızın pişkinliğine aldırmadan cevabını yapıştırır:\n‘Be adam ne kızıyorsun? Senin sırtındakiler bizim evin eşyaları değil mi? Ben de buraya yeni taşındığımızı sanıyordum!’"
  },
  {
    "prompt": "Buzağı İken Koştuğunu Gördüm",
    "story": "Günün birinde ciritçiler cirit oynamak için Hoca’yı meydana davet ederler. Hoca da at yerine bir öküze biner ve meydana varır. Hoca’nın bu hâlini gören herkes bir taraftan güler, bir taraftan da;\n‘Hocam, hiç öküz koşar mı, niye ata binmedin?’ deyince Hoca;\n‘Dostlar, niçin gülersiniz, ben bunun buzağı iken koştuğunu gördüm, onun için bununla geldim.’ der."
  },
  {
    "prompt": "Bülbül Derler",
    "story": "Birkaç şehirli dağda gezerlerken bir kirpi bulurlar. Bilmedikleri bu hayvanı torbalarına koydukları gibi Hoca’nın kapısını çalarlar:\n‘Hocam, biz böyle bir yaratık bulduk, buna ne derler?’\n‘Efendiler, ben bir araştırayım, bana bu gece izin verin, yarın gelin size cevap vereyim.’ der.\nBelirtilen saatte şehirliler gelince Hoca;\n‘Arkadaşlar, ben bunu araştırdım, buna kocaman bülbül derler.’ der."
  },
  {
    "prompt": "Cennet ile Cehennem Dolana Kadar",
    "story": "Geveze adamın biri Nasreddin Hoca’yla sokakta karşılaşır.\n‘Hoca Efendi, sen görmüş geçirmiş ve okumuş bir adamsın, bilirsin. İnsanlar ne zamana kadar ölecekler?’ diye sorar.\nHoca adamın niyetini anlamıştır, şöyle bir sakalını sıvazladıktan sonra;\n‘Be adam, bunu bilemeyecek ne var? Cennet ile cehennem dolana kadar.’ deyiverir."
  },
  {
    "prompt": "Ciğeri kedi yedi diyen Hoca baltayı saklar.",
    "story": "Nasreddin Hoca zaman zaman evine ciğer getirir. Fakat ne tuhaftır ki akşam sofrada ciğer kebabının yerine başka yemeklerle karşılaşır. Bir gün böyle, iki gün böyle derken Hoca dayanamaz ve hanımına sorar:\n“Yahu hatun, getirdiğim ciğerlere ne oldu?”\nHoca'nın hanımı hiçbir şey olmamışçasına;\n“Aman Hocam, sorma her defasında ciğerin kokusunu alan tekir, ben mutfağa girmeden yiyip bitiriyor.” der.\nBu sözleri işiten Hoca birdenbire yerinden kalkar ve köşedeki baltayı kaptığı gibi koşmaya başlar, bir süre sonra da hanımının yanına gelir:\n“Hoca baltayı nettin?”\n“Sakladım.”\n“Niçin?”\n“Kedi yemesin diye.”\nHoca’nın hanımı dayanamayıp itiraz eder.\n“Yahu Hocam, kedi baltayı yer mi?”\nHoca, hanımını şöyle bir süzdükten sonra cevabını verir:\n“Yer hanım yer, üç beş akçelik ciğeri yiyen kedi, acaba yüz akçelik baltayı yemez mi?”"
  },
  {
    "prompt": "Eşeği yavaş giden Hoca üç gün önceden cuma için yola çıkar.",
    "story": "Nasreddin Hoca günün birinde Akşehir’de pazarı dolaşmaya başlar. Bu arada komşu köylerin birinden birkaç köylü ile karşılaşır. Köylüler Hoca’ya; “Hoca Efendi, bir cuma vakti bizim köye kadar gelseniz de sizin arkanızda bir namaz kılsak!” derler. Hoca; “Neden olmasın, bu hafta geleyim!” der.\nHoca ertesi gün eşeğine binerek köyün yolunu tutar. Olacak bu ya, yolu üzerinde eski dostlarından biriyle karşılaşır. Tanıdığı sorar:\n“Hayırdır Hocam, nereye gidersin böyle?”\n“Filanca köye cuma namazı kıldırmaya gidiyorum.”\n“Ama Hocam, bugün günlerden salı. Cumaya daha üç gün var.”\nHoca cevap verir:\n“Vallahi komşu, sen bu eşeğin huyunu suyunu bilmezsin; ben bununla o köye cumaya kadar ancak giderim.”"
  },
  {
    "prompt": "Hanım tartışmasını cübbe ile açıklayan Hoca aslında içindedir.",
    "story": "Nasreddin Hoca şehre inmek için evinden çıkar. Bahçe kapısında bir komşusuyla karşılaşır. Komşusu:\n“Hocam, geceleyin sizin evden gürültüler geliyordu, merak ettim, hayrola?”\nHoca:\n“Bir şey yoktu, hatunla biraz tartışmıştık, demek ki ağız dalaşımızı duymuşsun.”\nKomşu ısrarla sorar:\n“Hocam, cübbeden hiç öyle ses çıkar mı?”\nHoca dayanamaz:\n“Yahu komşu, ne uzatıp duruyorsun, cübbenin içinde ben de vardım.”"
  },
  {
    "prompt": "Çaylak kuşunun cinsiyetini anlamak için bir yıl çaylak olun.",
    "story": "Birkaç kafadar Hoca’ya takılmak maksadıyla;\n“Hocam, çaylak kuşu için altı ay erkek, altı ay dişi olur diyorlar, acaba doğru mu?”\nHoca cevap verir:\n“Bakın çocuklar! Bunun doğru olup olmadığını anlayabilmek için sizin bir yıl çaylak olmayı denemeniz gerekir!”"
  },
  {
    "prompt": "Tuvalette sakız çiğnemez çünkü dışarıdan yanlış anlaşılır.",
    "story": "Bir adam Hoca’ya;\n“Hocam, helada sakız çiğnenir mi?”\nHoca cevap veremez, araştıracağını söyler. Döndüğünde der ki:\n“Efendi, kitapta yeri yok ama sen çiğnemesen iyi edersin.”\nAdam:\n“Hocam, neden çiğnemeyeyim?”\nHoca:\n“Sen tuvalette sakızı çiğnerken kapının dışındakiler senin başka şey çiğnediğini sanırlar.”"
  },
  {
    "prompt": "Hoca ve hanımı yolda, iki gün daha gitsek bitiyor der hanım, Hoca yarıladık der.",
    "story": "Nasreddin Hoca ile hanımı seyahate çıkarlar. Bir süre gittikten sonra Hoca sorar:\n“Hanım, daha ne kadar yolumuz var?”\nHanımı:\n“Bugün ve yarın da gidersek iki günlük yolumuz kaldı.”\nHoca sevinçle:\n“Desene hanım, daha şimdiden yolu yarıladık.”"
  },
  {
    "prompt": "Hoca damdan düşer, halinden sadece damdan düşen anlar.",
    "story": "Hoca evinin damında çalışırken, olacak bu ya, aşağıya düşüverir. Haberi duyan komşuları;\n\"Hocam, geçmiş olsun, damdan düşmüşsün, çok üzüldük.\" derler ve ardından soru üstüne soru sorarlar:\n\"Nasıl oldu?\"\n\"Neden dikkat etmedin?\"\n\"Bir daha dikkatli ol…\"\nSorular uzadıkça, Hoca’nın da canı sıkılmaya başlar. Düşünür, taşınır ve bunların hepsini birden susturmak için komşularına;\n\"Komşular, sizin içinizde damdan düşeniniz var mı?\" deyince, misafirler hep bir ağızdan;\n\"Yook…\" diye cevap verir. Bu defa Hoca;\n\"Öyleyse boşuna konuşmayın, benim hâlimden ancak damdan düşen anlar!\" der."
  },
  {
    "prompt": "Deniz suyu neden tuzludur?",
    "story": "Günün birinde Hoca'nın da içinde bulunduğu toplulukta yarenlik edilirken, hazır bulunanlardan biri Hoca'yı imtihan edercesine bir soru sorar:\n\"Hocam, denizlerin suyu niçin tuzludur?\"\n“Aaa, bunu bilmeyecek ne var, balıklar kokmasın diye.”"
  },
  {
    "prompt": "Rüyada 999 akçeyi az bulan Hoca, uyanınca fikrini değiştirir.",
    "story": "Hoca’ya rüyasında dokuz yüz doksan dokuz akçe verirler, ancak Hoca;\n\"Bin olmazsa kabul etmem.\" diye direnirken uyanmaz mı?\nElinin boş olduğunu gören Hoca tekrar yatar ve avuçlarını açarak;\n\"Verin, kabulümdür, dokuz yüz doksan dokuzu veren Allah birini de verir!\" deyiverir."
  },
  {
    "prompt": "Dünyanın merkezi eşeğin ayağının altıymış.",
    "story": "Günün birinde üç papazın yolu Akşehir’e uğrar. Burada Nasreddin Hoca ile sohbet eden papazlar, Efendi’nin bilgisini denemek isterler. İlk soruyu birinci papaz sorar:\n\"Hocam, dünyanın merkezi neresidir?\"\nHoca hiç tereddüt etmeden eşeğini göstererek;\n\"Eşeğimin sağ ön ayağını bastığı yerdir.\" diye cevap verir.\nİçlerinden biri itiraz eder:\n\"Bunu nereden biliyorsun?\"\n\"İnanmıyorsanız ölçün.\"\nBu defa ikinci papaz sorar:\n\"Hocam, gökte kaç yıldız vardır?\"\nHoca bu soruya da tereddüt etmeden yine eşeğini göstererek cevap verir:\n\"Gökyüzünde, eşeğimin kuyruğundaki kıl kadar yıldız vardır.\"\n\"Bunu ispatlayabilir misiniz?\" denildiğinde Nasreddin Hoca;\n\"Arzu ederseniz sayabilirsiniz.\" der.\nHoca’nın sorulan sorulara verdiği cevaplar, papazları şaşırtınca üçüncü soruyu sormaktan vazgeçerler."
  },
  {
    "prompt": "Elin eşeği türkülerle aranır.",
    "story": "Bir gün subaşının eşeği kaybolur. Hoca, birkaç komşusu ile birlikte eşeği aramaya çıkar. Hoca hem eşeği aramakta hem de türkü söylemektedir. Bu durumu yadırgayan komşularından biri Hoca’ya;\n\"Hocam, bu nasıl iş, insan kaybolan eşeği böyle türkü söyleyerek mi arar?\" diye sorar.\nHoca bu lafın altında kalır mı?\n\"El elin eşeğini türkü çağıra çağıra arar.\" der."
  },
  {
    "prompt": "Hoca oğlunun doğumuna şükreder, müjdeciyi boşverir.",
    "story": "Günün birinde Hoca’nın bir çocuğu olur. Bu sırada Hoca da bir yolculuktan dönmektedir. Komşularından birisi Hoca’yı karşılar;\n\"Hoca Efendi, oğlun oldu; müjdemi ver.\" der.\nHaberi alan Hoca;\n\"Çok şükür ya Rabbi.\" diye karşılık verir.\nBunun üzerine komşusu;\n\"Hocam, şükredeceğine müjdemi versen.\" deyince o da;\n\"Yahu komşu, doğduysa benim çocuğum doğdu, sana ne, elbette şükredeceğim.\" der."
  },
  {
    "prompt": "Gölden abdest alırken yön meselesi.",
    "story": "Lüzumsuz adamın birisi Hoca’yı sıkıştırmak için bir soru sorar:\n\"Hocam, gölde abdest alırken hangi yöne dönmeliyim?\"\nBu soru üzerine Hoca gülümser ve;\n\"Elbiselerin hangi tarafta ise oraya dön!\" deyiverir."
  },
  {
    "prompt": "Hoca, herkesin sözlerine kulak asmamak gerektiğini anlatır.",
    "story": "Günün birinde Nasreddin Hoca ile oğlunun komşu köylerden birine işleri düşer. Birlikte yola çıkarlar. Yolculuk sırasında Hoca, küçük olduğu için önce oğlunu eşeğe bindirir. Biraz sonra karşılarına çıkan bir adam, eşek ve üstündeki çocuğu iyice bir süzdükten sonra;\n\"Hey gidi zamane gençleri hey! Hiç utanmadan kendileri eşeğe binerler, yaşlı, bilgin babalarını yürütürler!\" diye söylenir.\nAdam, yanlarından geçip giderken oğul da utancından kıpkırmızı olur, eşekten iner ve babasını bindirir. Biraz sonra karşılaştıkları adamlar da başlarlar söylenmeye:\n\"Aman, şuna da bak! Senin yaşın geçmiş, kemiğin kartlaşmış; hem işte geldin, işte gidiyorsun. Şu taze fidanı eşeğe bindir de yorma zavallıyı!\"\nBu söz üzerine Hoca Efendi oğlunu da eşeğe bindirir ve baba oğul eşeğin üstünde yollarına devam ederler.\nBir süre bu şekilde yol aldıktan sonra birkaç kişi daha karşılarına gelir. Bunlar da başlarlar konuşmaya:\n\"Amma acımasız adamlar var şu dünyada!\"\n\"Bu zavallı eşek ikinizi nasıl taşısın?\"\nBu söz üzerine Hoca Efendi ve oğlu eşekten inerler. Eşeği önlerine katarak kırıta kırıta giderlerken karşılaştıkları adamlar da bu duruma karışmadan duramazlar:\n\"Allah Allah, bu ne budalalık yahu!\"\n\"Bak yahu, eşek önlerinde bomboş, hoplaya zıplaya keyifle gidiyor.\"\nBütün bunları duyan Hoca, adamlar uzaklaştıktan sonra oğluna der ki:\n\"Bak oğul, adamları gördün işte… Hiçbirini memnun edemedik… Ne yapalım elin ağzı torba değil ki büzesin.\""
  },
  {
    "prompt": "Hanım ipe un serdiği için ip verilemedi.",
    "story": "Günün birinde komşularından biri Nasreddin Hoca’dan çamaşır ipini ister. Komşunun tavrı Nasreddin Hoca’nın hiç hoşuna gitmez, çünkü komşu aldığı emaneti geri vermeyen biridir. Hoca;\n\"Komşucuğum, biraz bekle; ben ipi bulayım.\" der.\nBir süre sonra Hoca kapıda görünür.\n\"Vallahi komşum, bizim hanım ipe un sermiş.\"\nBu cevaba şaşıran komşu kızgınlığını gizleyemez ve;\n\"Yahu Hoca Efendi; alay mı ediyorsun sen, hiç ipe un serilir mi?\" der.\nHoca adamı umursamayan bir tavırla cevap verir:\n\"Ee!. . İnsanın canı vermek istemeyince ipine un da serer, buğday da…\""
  },
  {
    "prompt": "Baş ağrısına çözüm olarak diş çektirme önerisi.",
    "story": "Günün birinde pazardan dönmekte olan Hoca'nın önünü bir komşusu keser ve derdini bir bir anlatır. Hoca onu biraz oyalamak isteyince komşusu tekrar;\n\"Ama Hocam, başım çok ağrıyor.\" der. Hoca şöyle sağına soluna baktıktan sonra, düşünür gibi yapar ve ardından cevabını veriverir:\n\"Bak komşu, senin derdinin dermanını şimdi hatırladım. Bundan birkaç hafta önce benim de dişim ağrımıştı, epeyce direndikten sonra baktım olacak gibi değil, gittim dişçiye, dişimi çektirdim. Meğer başımın ağrısının dermanı buymuş. Haydi git sen de dişini çektir.”"
  },
  {
    "prompt": "Nasreddin Hoca’ya göre hekimliğin tanımı.",
    "story": "Nasreddin Hoca’ya sormuşlar:\n\"Hekimlik nedir?\"\nO da en güzel cevabı vermiş:\n\"Ayağını sıcak tut, başını serin; gönlünü ferah tut, düşünme derin derin.\""
  },
  {
    "prompt": "Hınk demenin bedeli duvara çarpan akçe ile ödenir.",
    "story": "Hoca’nın kadılık yaptığı yıllarda iki kişi birbirinden davacı olur. Bir süre sonra da Hoca’nın huzuruna gelirler. Hoca’nın;\n\"Derdiniz nedir, anlatın bakayım.\" demesi üzerine, adamlardan biri bağırıp çağırarak konuşur:\n\"Kadı Efendi, bu adamdan davacıyım.\"\nHoca adamı sakin olmaya davet eder ve;\n\"Önce sakin ol ve derdini anlat da bir dinleyeyim.\" der.\n\"Bu adam ormanda odun keserken ben de onun yanındaydım. O baltayı ağaca her vurduğunda ben de ‘hınk’ diyerek yardımcı oluyordum. Adam ağaçları kesti kesmesine de ben paramı alamadım. Söyle buna Kadı Efendi, ödesin borcunu.\"\nHoca adamları iyice süzdükten sonra iyi birine benzettiği oduncuya döner:\n\"Sen bu adamın borcunu niye ödemedin?\"\n\"Aman Kadı Efendi, ben ona, ‘Hınk mınk de’ demedim. Sonra hınk demenin bedeli mi olurmuş?\"\nHoca her ikisini de dinledikten sonra kendine özgü yöntemle adaleti dağıtmaya karar verir.\n\"Olur, olur, bal gibi olur. Şimdi sen bu adama on akçelik borcunu öde bakalım.\"\nOduncu şaşırır ama Kadı’ya da bir şey diyemez. Çıkarır on akçeyi Kadı’ya verir. Kadı madeni paraları duvara çarpınca sesler çıkmaya başlar. Bu sırada Hoca da ‘hınk’ların bedelini isteyen adama dönerek;\n\"İşte, aldın hınklarının bedelini, haydi şimdi gidin.\" der.\n\"Kadı Efendi, Kadı Efendi! Sen beni aldatıyorsun. İki ses çıkardın, bizim para ne oldu?\"\nKadı, parayı oduncuya teslim ederken, öbürüne;\n\"Uzatma adam! Senin hınklarının bedeli de ancak on akçenin sesiyle ödenir.\" der."
  },
  {
    "prompt": "Hırsızın da suçu olduğunu Hoca hatırlatır.",
    "story": "Bir yaz gecesi Hoca ile hanımı sıcağa dayanamadıkları için damda yatmaya karar verirler. Herkesin derin uykuya daldığı sırada hırsızlar Hoca’nın evine girerler ve buldukları her şeyi aldıkları gibi giderler.\nSabahleyin aşağıya, evine inen Hoca, eşyalarının çalındığını görünce, kapıya çıkarak bağırmaya başlar:\n\"Yetişin komşular, evimize hırsız girmiş; her şeyimizi çalmışlar!\"\nHoca’nın sesini duyan komşuları, onun yanına gelirler ve arka arkaya sorular sormaya başlarlar:\n\"Ah Hocam, ah! Hiç insan geceleyin damda yatar mı?\"\n\"Hocam, kapının arkasına sürgüsünü takmamış mıydın?\"\n\"Hocam, kilit bozuk muydu yoksa?\"\nHoca bakar ki soruların ardı arkası kesilmeyecek, dayanamaz;\n\"Bre komşular, doğru söylüyorsunuz da, bizim hırsızın hiç mi suçu yok?\" der."
  },
  {
    "prompt": "Ayaklarını kaybeden çocuklara Hoca bastonla çözüm bulur.",
    "story": "Sıcak bir yaz gününde serinlemek için ayaklarını Akşehir Gölü’ne sokan çocuklar bir süre sonra ‘ayaklarımız karıştı’ diye kavga etmeye başlarlar. Hatta en küçük çocuk; ‘ayaklarımı kaybettim’ diye ağlamaya başlar.\nO sırada oradan geçmekte olan Nasreddin Hoca, gürültünün olduğu tarafa doğru yönelir ve;\n\"Çocuklar, hayırdır, nedir bu gürültü?\" diye sorar. İçlerinden biri;\n\"Hocam, Hocam, ayaklarımız karıştı, bunları nasıl ayıracağız?\" der ve ardından ağlamaya başlar. Bunun üzerine Hoca;\n\"Çocuklarım, ağlamayın, ben şimdi sizin ayaklarınızı bulurum.\" der ve hemen ardından bastonunu suya daldırır. Sonra da çocukların ayaklarına vurmaya başlar. Ayakları acıyan çocuklar da \"Buldum\" diyerek ayaklarını dışarıya çıkarırlar.\nDayağın korkusuyla diğer çocuklar da ayaklarını sudan çıkarınca Hoca, sıkıntıyı çözmüş olmanın verdiği keyifle yoluna devam eder."
  },
  {
    "prompt": "Nasreddin Hoca’ya aşk sorusu ve esprili cevabı.",
    "story": "Bir gün Hoca ve öğrencileri ders sırasında sohbet ederlerken, muzibin biri Hoca’ya sorar:\n\"Hocam!\"\n\"Buyur evladım.\"\n\"Hocam, hayatınızda hiç âşık oldunuz mu?\"\nSoru Hoca’nın hoşuna gider gitmesine de, bazı aile sırlarının da açığa çıkmasına gönlü razı değildir. Bu sebepten düşünür, taşınır ve;\n\"Çocuğum, bir keresinde tam âşık olacaktım, o sırada üzerime geldiler.\" deyiverir."
  }
  
  

]


with open("fikra_dataset.json", "w", encoding="utf-8") as f:
    json.dump(fikralar, f, ensure_ascii=False, indent=2)

print("✅ Fıkra dataseti başarıyla oluşturuldu: fikra_dataset.json")


