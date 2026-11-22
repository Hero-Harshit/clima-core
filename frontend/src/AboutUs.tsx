import { X } from "lucide-react";

type AboutUsProps = {
  onClose: () => void;
};

function AboutUs({ onClose }: AboutUsProps) {
  return (
    <section className="provenance-modal about-modal" role="dialog" aria-modal="true" aria-labelledby="about-title">
      <header className="drawer-header">
        <div>
          <span className="section-kicker">ABOUT CLIMA-CORE</span>
          <h2 id="about-title">About Clima-Core</h2>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close About Clima-Core"><X size={18} /></button>
      </header>

      <article className="about-content">
        <section className="about-section about-introduction">
          <h3>Building Intelligence for a Changing Environment</h3>
          <p>Environmental challenges are becoming increasingly interconnected. Extreme temperatures, unpredictable rainfall, air pollution, water stress, and natural disasters affect communities, infrastructure, agriculture, and ecosystems.</p>
          <p>Understanding these challenges requires more than isolated observations. It requires the ability to analyse environmental data, identify emerging patterns, assess potential risks, and transform complex information into meaningful insights.</p>
          <p><strong>Clima-Core is an AI-powered Environmental Intelligence and Decision Support Platform designed to make this possible.</strong></p>
          <p>By bringing together machine learning, scientific modelling, statistical analysis, and geospatial intelligence, Clima-Core aims to provide a more integrated understanding of environmental conditions and their potential impacts.</p>
          <p>Our vision is to make environmental intelligence more accessible, interpretable, and useful for the people and organisations responsible for making informed decisions.</p>
        </section>

        <section className="about-section">
          <h3>Our Mission</h3>
          <p>Our mission is to transform complex environmental and climate-related data into meaningful intelligence that supports better awareness, preparedness, and decision-making.</p>
          <p>We aim to bridge the gap between scientific data and practical applications by combining predictive technologies with physical models and geographical context.</p>
          <p>Rather than treating environmental hazards as isolated events, Clima-Core explores how different environmental factors interact and how their combined effects may influence people, infrastructure, and natural systems.</p>
        </section>

        <section className="about-section">
          <h3>What We Do</h3>
          <p>Clima-Core integrates multiple sources of environmental information and analytical technologies into a unified platform.</p>
          <p>Our work focuses on several important areas.</p>

          <h4>1. Environmental Risk Intelligence</h4>
          <p>We analyse environmental conditions to identify and estimate potential risks associated with extreme weather and environmental stress.</p>
          <p>This includes:</p>
          <ul>
            <li>Flood and rainfall risk.</li>
            <li>Extreme heat and thermal stress.</li>
            <li>Air quality and atmospheric pollution.</li>
            <li>Drought and soil moisture conditions.</li>
            <li>Water quality and contamination indicators.</li>
            <li>Agricultural and crop-related environmental stress.</li>
            <li>Electricity demand and environmental stress interactions.</li>
            <li>Compound environmental hazards.</li>
          </ul>

          <h4>2. Predictive Intelligence</h4>
          <p>Clima-Core uses machine learning and statistical techniques to analyse historical observations and relevant environmental variables.</p>
          <p>Our predictive models are developed to estimate environmental conditions, identify patterns, and support forward-looking assessments.</p>
          <p>Different modelling approaches are used depending on the nature of the problem, available data, and required level of prediction.</p>

          <h4>3. Physical and Scientific Modelling</h4>
          <p>Machine learning is only one part of environmental intelligence.</p>
          <p>Clima-Core also incorporates scientific and physical modelling approaches to represent environmental processes.</p>
          <p>These include hydrological simulations, rainfall-runoff modelling, biometeorological calculations, extreme-value analysis, drought-regime modelling, and statistical dependence analysis.</p>
          <p>Combining predictive models with physical calculations helps provide additional scientific context to environmental assessments.</p>

          <h4>4. Geospatial and Exposure Intelligence</h4>
          <p>Environmental risks are closely connected to geography.</p>
          <p>Clima-Core uses geographical information and spatial datasets to investigate how environmental conditions relate to surrounding populations, infrastructure, river networks, terrain, and other important geographical features.</p>
          <p>This helps move beyond isolated environmental measurements towards a broader understanding of geographical exposure and potential vulnerability.</p>

          <h4>5. Decision Support</h4>
          <p>Environmental information becomes more useful when it can be interpreted in the context of real-world responsibilities.</p>
          <p>Clima-Core is designed to support different stakeholders, including disaster-management professionals, municipal authorities, environmental researchers, infrastructure planners, agricultural stakeholders, public-health professionals, and citizens.</p>
          <p>Its decision-support capabilities aim to translate complex analytical outputs into understandable insights relevant to different areas of responsibility.</p>
        </section>

        <section className="about-section">
          <h3>How Clima-Core Works</h3>
          <p>Clima-Core follows a modular four-layer architecture. Each layer performs a distinct role within the environmental intelligence pipeline.</p>
          <h4>Layer 1: Predictive Intelligence</h4>
          <p>This layer contains machine learning and statistical models that analyse environmental variables and generate predictions or risk estimates.</p>
          <h4>Layer 2: Physical and Statistical Intelligence</h4>
          <p>This layer applies scientific equations, physical simulations, statistical methods, and environmental calculations to interpret and complement model outputs.</p>
          <h4>Layer 3: Exposure Intelligence</h4>
          <p>This layer combines environmental information with geospatial datasets to assess the geographical relationship between hazards, populations, infrastructure, and natural systems.</p>
          <h4>Layer 4: Decision Intelligence</h4>
          <p>This layer translates the combined information into contextual insights and decision-support guidance for different stakeholder groups.</p>
          <p>Together, these layers form an integrated framework for understanding environmental conditions and their potential consequences.</p>
        </section>

        <section className="about-section">
          <h3>Technology Behind Clima-Core</h3>
          <p>Clima-Core brings together multiple technologies and scientific disciplines.</p>
          <p>Our technical foundation includes:</p>
          <ul>
            <li><strong>Machine Learning:</strong> Gradient boosting, regression, classification, ensemble learning, and specialised neural architectures.</li>
            <li><strong>Statistical Modelling:</strong> Probabilistic forecasting, uncertainty calibration, copula modelling, and extreme-value statistics.</li>
            <li><strong>Hydrological Modelling:</strong> Rainfall-runoff simulation, streamflow analysis, catchment modelling, and water-balance calculations.</li>
            <li><strong>Atmospheric Science:</strong> Weather forecast processing, thermal analysis, rainfall assessment, and air-quality modelling.</li>
            <li><strong>Geospatial Intelligence:</strong> Terrain analysis, river network processing, population exposure, and infrastructure mapping.</li>
            <li><strong>Explainable AI:</strong> Model interpretation and feature attribution to help understand the factors influencing predictions.</li>
          </ul>
          <p>Clima-Core also draws on environmental datasets and scientific resources from recognised research institutions, government agencies, and international data providers, subject to their respective source terms and usage conditions.</p>
        </section>

        <section className="about-section">
          <h3>Who We Aim to Serve</h3>
          <p>Clima-Core is being developed with a broad range of potential users in mind.</p>
          <div className="about-audience">
            <div><h4>Government and Disaster Management Authorities</h4><p>To support environmental risk awareness, geographical assessments, preparedness, and planning activities.</p></div>
            <div><h4>Municipal Corporations and Urban Planners</h4><p>To help understand environmental pressures affecting urban areas, infrastructure, and communities.</p></div>
            <div><h4>Environmental Researchers and Analysts</h4><p>To explore historical environmental patterns, predictive models, physical simulations, and statistical relationships.</p></div>
            <div><h4>Agricultural Stakeholders</h4><p>To provide climate-related and environmental insights relevant to crop suitability, agricultural stress, and seasonal planning.</p></div>
            <div><h4>Infrastructure and Utility Professionals</h4><p>To support the assessment of environmental conditions that may influence electricity systems, transportation networks, and essential services.</p></div>
            <div><h4>Citizens and Communities</h4><p>To make environmental information easier to understand and help people become more aware of conditions affecting their surroundings.</p></div>
          </div>
        </section>

        <section className="about-section">
          <h3>Our Approach to Responsible Environmental Intelligence</h3>
          <p>We believe that technological innovation must be supported by scientific responsibility.</p>
          <p>Environmental predictions are inherently uncertain. Data coverage, observation quality, geographical resolution, and changing environmental conditions can influence model performance.</p>
          <p>For this reason, Clima-Core aims to maintain several important principles:</p>
          <ul>
            <li><strong>Scientific Responsibility:</strong> Use appropriate scientific methods and distinguish physical calculations from statistical predictions.</li>
            <li><strong>Data Transparency:</strong> Recognise and document the sources and limitations of environmental information.</li>
            <li><strong>Explainability:</strong> Make analytical outputs easier to interpret and understand.</li>
            <li><strong>Uncertainty Awareness:</strong> Communicate uncertainty wherever meaningful rather than presenting estimates as absolute facts.</li>
            <li><strong>Responsible AI:</strong> Use automated systems as decision-support tools, not as unquestionable authorities.</li>
            <li><strong>Continuous Improvement:</strong> Evaluate models, examine their limitations, and improve the system through reproducible research and validation.</li>
          </ul>
          <p>Clima-Core is not a replacement for official meteorological warnings, government emergency communications, professional environmental assessments, or emergency-response authorities.</p>
          <p>Its purpose is to complement existing information systems by providing additional environmental analysis and decision-support capabilities.</p>
        </section>

        <section className="about-section">
          <h3>Our Vision</h3>
          <p>We envision a future where environmental intelligence is not limited to isolated datasets, specialised research institutions, or disconnected analytical systems.</p>
          <p>We want to contribute towards a more connected approach in which environmental observations, predictive modelling, physical science, and geographical intelligence work together.</p>
          <p>As environmental challenges become more interconnected, the ability to understand their relationships and potential impacts becomes increasingly important.</p>
          <p><strong>Our vision is simple: to turn environmental data into meaningful intelligence that helps people understand risks, anticipate challenges, and make better-informed decisions.</strong></p>
        </section>

        <section className="about-section">
          <h3>Towards a More Informed Future</h3>
          <p>Clima-Core represents an ongoing effort to bring artificial intelligence, environmental science, and geospatial technology together.</p>
          <p>We believe that better environmental understanding begins with better information, stronger scientific foundations, and responsible use of technology.</p>
          <p>We are continuously working towards making environmental intelligence more comprehensive, accessible, and useful.</p>
          <p><strong>Clima-Core. Intelligence for a changing environment.</strong></p>
        </section>

        <div className="about-contact">
          <p><strong>Project:</strong> Clima-Core Environmental Intelligence Platform</p>
          <p><strong>Website:</strong> https://climacore.ai</p>
          <p><strong>Contact:</strong> contact@climacore.ai</p>
          <p><strong>Initiative:</strong> Clima-Core Open Science Initiative</p>
        </div>
      </article>
    </section>
  );
}

export default AboutUs;
