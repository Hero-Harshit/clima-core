import { LockKeyhole, X } from "lucide-react";

type PrivacyPolicyProps = {
  onClose: () => void;
};

function PrivacyPolicy({ onClose }: PrivacyPolicyProps) {
  return (
    <section className="provenance-modal privacy-modal" role="dialog" aria-modal="true" aria-labelledby="privacy-policy-title">
      <header className="drawer-header">
        <div>
          <span className="section-kicker">PRIVACY & DATA</span>
          <h2 id="privacy-policy-title">Privacy Policy</h2>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close privacy policy"><X size={18} /></button>
      </header>
      <div className="privacy-document">
        <div className="privacy-meta">
          <span><strong>Effective Date:</strong> 1 November 2026</span>
          <span><strong>Last Updated:</strong> 22 November 2026</span>
        </div>

        <section className="privacy-section">
          <h3>1. Introduction</h3>
          <p>Welcome to <strong>Clima-Core</strong>.</p>
          <p>At Clima-Core, we recognise the importance of protecting your privacy and handling information responsibly. This Privacy Policy explains how we collect, use, process, store, and protect information when you visit or interact with our website and its environmental intelligence services.</p>
          <p>Clima-Core is an AI-powered environmental intelligence platform designed to provide weather insights, environmental risk assessments, climate-related predictions, geospatial analysis, and decision-support information.</p>
          <p>By accessing or using Clima-Core, you acknowledge that you have read this Privacy Policy. Please review it carefully to understand our data practices.</p>
        </section>

        <section className="privacy-section">
          <h3>2. Information We Collect</h3>
          <p>We aim to collect only the information reasonably necessary to operate, maintain, and improve our services.</p>
          <p>Depending on the features you use, the following categories of information may be processed.</p>
          <h4>2.1 Information You Provide</h4>
          <p>When interacting with Clima-Core, you may voluntarily provide information such as:</p>
          <ul>
            <li>City, district, state, or other geographical locations.</li>
            <li>Search queries and environmental assessment requests.</li>
            <li>Preferences relating to environmental information or decision-support features.</li>
            <li>Information submitted through contact forms or feedback channels, if available.</li>
          </ul>
          <p>Please avoid submitting sensitive personal information unless a feature specifically requires it.</p>
          <h4>2.2 Location Information</h4>
          <p>Clima-Core uses geographical information to provide location-specific environmental insights.</p>
          <p>This may include:</p>
          <ul>
            <li>Locations manually entered by users.</li>
            <li>Selected cities, districts, or geographical areas.</li>
            <li>Coordinates provided as part of an environmental assessment request.</li>
            <li>Device location information, only where a location-enabled feature requests and receives the relevant permission.</li>
          </ul>
          <p>Location information is used to identify the relevant geographical area for environmental analysis and to provide location-specific results.</p>
          <p>You may choose to enter a location manually instead of using a device location feature, where that option is available.</p>
          <h4>2.3 Technical and Usage Information</h4>
          <p>When you access our website, certain technical information may be collected automatically by our website infrastructure or security systems, depending on the services and configurations in use.</p>
          <p>This may include:</p>
          <ul>
            <li>IP address.</li>
            <li>Browser type and version.</li>
            <li>Device type and operating system.</li>
            <li>General usage information.</li>
            <li>Access timestamps.</li>
            <li>Requested pages and features.</li>
            <li>Error reports and diagnostic information.</li>
          </ul>
          <p>This information may be used to maintain website security, identify technical issues, monitor service availability, and improve performance.</p>
          <h4>2.4 Cookies and Similar Technologies</h4>
          <p>Clima-Core may use essential cookies or similar technologies to support website functionality, maintain user preferences, improve security, and understand basic website performance.</p>
          <p>Where analytics or optional tracking technologies are enabled, their use and applicable consent choices will be explained through the relevant website controls.</p>
          <p>You may manage or restrict cookies through your browser settings. Disabling certain cookies may affect some website features.</p>
        </section>

        <section className="privacy-section">
          <h3>3. Environmental and Scientific Data</h3>
          <p>Clima-Core processes environmental and scientific information from various meteorological, hydrological, geospatial, agricultural, and environmental data sources.</p>
          <p>These may include information relating to:</p>
          <ul>
            <li>Rainfall and temperature.</li>
            <li>Weather forecasts and atmospheric conditions.</li>
            <li>River networks and catchment characteristics.</li>
            <li>Soil moisture and groundwater quality.</li>
            <li>Air pollution and atmospheric composition.</li>
            <li>Population distribution and infrastructure.</li>
            <li>Agricultural and climatic conditions.</li>
            <li>Historical environmental observations.</li>
          </ul>
          <p>Such datasets are used for scientific analysis, predictive modelling, environmental risk assessment, and geographical exposure estimation.</p>
          <p>Environmental observations and publicly available scientific datasets are generally not information about an individual user. However, a geographical location submitted by a user may constitute personal data when it can be associated with that individual.</p>
        </section>

        <section className="privacy-section">
          <h3>4. How We Use Information</h3>
          <p>Information processed through Clima-Core may be used for the following purposes:</p>
          <ol>
            <li><strong>Providing Environmental Intelligence:</strong> To generate location-specific weather information, environmental assessments, and climate-related insights.</li>
            <li><strong>Predictive Analysis:</strong> To process user requests through relevant machine learning models, statistical methods, and physical modelling engines.</li>
            <li><strong>Geospatial Analysis:</strong> To evaluate geographical conditions, population exposure, infrastructure proximity, and environmental susceptibility.</li>
            <li><strong>Decision Support:</strong> To generate contextual environmental information and recommendations for different user categories.</li>
            <li><strong>Website Operations:</strong> To maintain website functionality, performance, reliability, and availability.</li>
            <li><strong>Security and Protection:</strong> To detect misuse, investigate technical issues, and protect the platform against unauthorised access.</li>
            <li><strong>Service Improvement:</strong> To evaluate system performance and improve the accuracy, reliability, and usability of Clima-Core.</li>
          </ol>
          <p>We do not intend to use personal information for purposes unrelated to the services described in this Policy without an appropriate legal basis.</p>
        </section>

        <section className="privacy-section">
          <h3>5. Artificial Intelligence and Automated Processing</h3>
          <p>Clima-Core uses machine learning algorithms, statistical models, physical calculations, and automated analytical processes to generate environmental insights.</p>
          <p>Information submitted through the platform may be processed by these systems to produce predictions, risk estimates, geographical assessments, and scenario-based outputs.</p>
          <p>Important considerations:</p>
          <ul>
            <li>Automated predictions are generated using available data and model configurations.</li>
            <li>Results may vary depending on geographical coverage, data availability, and environmental conditions.</li>
            <li>Some outputs may be estimates or simulated scenarios rather than direct observations.</li>
            <li>We do not represent every model-generated result as a confirmed real-world event.</li>
            <li>Personal information is not intentionally used to identify individuals through environmental prediction models.</li>
          </ul>
          <p>Clima-Core is designed as an environmental intelligence and decision-support platform. Its automated outputs are not a substitute for official government warnings, professional assessments, or emergency instructions.</p>
        </section>

        <section className="privacy-section">
          <h3>6. Data Storage and Retention</h3>
          <p>Information may be processed or stored through the infrastructure and service providers used to operate Clima-Core.</p>
          <p>We retain information only for as long as reasonably necessary to fulfil the purposes described in this Policy, maintain platform functionality, meet applicable legal requirements, and address legitimate operational or security needs.</p>
          <p>Retention periods may vary depending on the type of information and the purpose for which it is processed.</p>
          <p>Where information is no longer required, we will take appropriate steps to delete, anonymise, or securely dispose of it, subject to applicable legal requirements.</p>
        </section>

        <section className="privacy-section">
          <h3>7. Data Sharing and Third-Party Services</h3>
          <p>Clima-Core does not intend to make users' personal information publicly available.</p>
          <p>Information may be shared with or processed by third parties only where reasonably necessary for operating the platform or where otherwise permitted or required by applicable law.</p>
          <p>These parties may include:</p>
          <ul>
            <li>Website hosting and infrastructure providers.</li>
            <li>Database and data storage providers.</li>
            <li>Security and performance monitoring providers.</li>
            <li>Technical service providers supporting website functionality.</li>
            <li>Government authorities or regulatory bodies where disclosure is legally required.</li>
          </ul>
          <p>Where third-party providers process personal information on our behalf, we seek to ensure that appropriate safeguards and contractual protections are in place.</p>
          <p>We do not authorise service providers to use personal information for purposes unrelated to the services they provide to Clima-Core.</p>
          <p>The environmental datasets referenced by Clima-Core may also be subject to their respective source providers' terms, attribution requirements, and licensing conditions.</p>
        </section>

        <section className="privacy-section">
          <h3>8. International Data Processing</h3>
          <p>Depending on the infrastructure and third-party services used, information may be processed or stored on servers located outside your state or country.</p>
          <p>Where such processing occurs, we will take appropriate measures in accordance with applicable data protection requirements.</p>
          <p>The location of servers and the applicable data protection arrangements may depend on the infrastructure providers supporting the platform.</p>
        </section>

        <section className="privacy-section">
          <h3>9. Data Security</h3>
          <p>We take reasonable technical and organisational measures to protect information processed through Clima-Core.</p>
          <p>These measures may include:</p>
          <ul>
            <li>Access restrictions and authentication controls.</li>
            <li>Secure communication protocols.</li>
            <li>Appropriate database access controls.</li>
            <li>Protection of sensitive configuration information.</li>
            <li>System monitoring and error detection.</li>
            <li>Software maintenance and security updates.</li>
            <li>Appropriate safeguards for stored information.</li>
          </ul>
          <p>Although we make reasonable efforts to protect information, no website, online service, or electronic storage system can guarantee absolute security.</p>
          <p>Users should avoid sharing confidential or sensitive personal information through features that do not specifically require it.</p>
        </section>

        <section className="privacy-section">
          <h3>10. Your Privacy Rights</h3>
          <p>Depending on applicable laws and the circumstances of processing, you may have rights concerning your personal information.</p>
          <p>These may include:</p>
          <ul>
            <li>Requesting information about the personal data processed by us.</li>
            <li>Requesting correction of inaccurate or incomplete information.</li>
            <li>Requesting deletion of personal information where legally applicable.</li>
            <li>Withdrawing consent where processing is based on consent.</li>
            <li>Raising a concern or complaint regarding the handling of your personal information.</li>
            <li>Exercising other rights available under applicable data protection laws.</li>
          </ul>
          <p>To submit a privacy-related request, please contact us using the details provided in the Contact Us section.</p>
          <p>We may request reasonable information to verify your identity before processing a request.</p>
          <p>Certain information may need to be retained where required by law or for legitimate security and operational purposes.</p>
        </section>

        <section className="privacy-section">
          <h3>11. Children's Privacy</h3>
          <p>Clima-Core is not intentionally designed to collect personal information from children without appropriate authorisation and safeguards.</p>
          <p>We do not knowingly seek to collect personal information from children in circumstances where such collection would violate applicable laws.</p>
          <p>If you believe that a child has submitted personal information to us without the required authorisation, please contact us so that the matter can be reviewed and appropriate action can be taken.</p>
        </section>

        <section className="privacy-section">
          <h3>12. Third-Party Websites and External Links</h3>
          <p>Our website may contain links to external websites, scientific resources, government portals, research publications, and third-party services.</p>
          <p>These websites operate independently and may have their own privacy policies and data practices.</p>
          <p>Clima-Core is not responsible for the privacy practices, content, or security of external websites.</p>
          <p>We encourage users to review the privacy policies of third-party websites before providing them with personal information.</p>
        </section>

        <section className="privacy-section">
          <h3>13. Changes to This Privacy Policy</h3>
          <p>We may update this Privacy Policy from time to time to reflect changes in our services, technology, data practices, or applicable legal requirements.</p>
          <p>When changes are made, the updated version will be published on this page with a revised Last Updated date.</p>
          <p>We encourage users to review this page periodically to remain informed about how Clima-Core handles information.</p>
        </section>

        <section className="privacy-section">
          <h3>14. Applicable Privacy Laws</h3>
          <p>Clima-Core aims to handle personal information responsibly and in accordance with applicable data protection and privacy laws.</p>
          <p>Where applicable, this includes the Digital Personal Data Protection Act, 2023 of India, the rules made under it, and other relevant legal requirements.</p>
          <p>The application of particular legal obligations depends on their commencement, applicability, and the nature of the services and information processing involved.</p>
        </section>

        <section className="privacy-section">
          <h3>15. Contact Us</h3>
          <p>If you have questions, concerns, or requests relating to this Privacy Policy or the handling of your personal information, please contact us.</p>
          <div className="privacy-contact">
            <p><strong>Project:</strong> Clima-Core Environmental Intelligence</p>
            <p><strong>Website:</strong> https://climacore.ai</p>
            <p><strong>Privacy Contact:</strong> privacy@climacore.ai</p>
            <p><strong>Organization:</strong> Clima-Core Open Science and Environmental Intelligence Initiative</p>
            <p><strong>Address:</strong> Technology Innovation Hub, New Delhi, India 110016</p>
          </div>
          <p>We will review privacy-related enquiries and respond through the appropriate communication channel.</p>
        </section>

        <div className="privacy-thanks">
          <LockKeyhole size={18} />
          <div>
            <strong>Thank you for using Clima-Core.</strong>
            <p>Your trust matters to us. We are committed to developing environmental intelligence responsibly while respecting the privacy and security of everyone who interacts with our platform.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default PrivacyPolicy;
